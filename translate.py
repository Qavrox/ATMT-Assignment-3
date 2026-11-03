import os
import glob
import logging
import argparse
import time
import numpy as np
import pickle
import sacrebleu
from tqdm import tqdm

import torch
import sentencepiece as spm
from torch.serialization import default_restore_location

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.decode import beam_search_decode, decode
from seq2seq.data.tokenizer import BPETokenizer
from seq2seq import models, utils
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler

def decode_to_string(tokenizer, array):
    """
    Takes a tensor of token IDs and decodes it back into a string."""
    if torch.is_tensor(array) and array.dim() == 2:
        return '\n'.join(decode_to_string(tokenizer, t) for t in array)
    return tokenizer.Decode(array.tolist())

def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--input', help='Path to the raw text file to translate (one sentence per line)')
    parser.add_argument('--data', help='Path to a prepared data directory or a prepared source split')
    parser.add_argument('--dicts', help='Directory containing SentencePiece tokenizer model files')
    parser.add_argument('--prepared-split', default='test', help='Split name to translate when --data is a directory')
    parser.add_argument('--src-tokenizer', help='path to source sentencepiece tokenizer')
    parser.add_argument('--tgt-tokenizer', help='path to target sentencepiece tokenizer')
    parser.add_argument('--checkpoint-path', required=True, help='path to the model file')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', required=True, type=str, help='path to the output file destination')
    parser.add_argument('--max-len', default=128, type=int, help='maximum length of generated sequence')
    # Beam search decoding parameters
    parser.add_argument('--beam-size', default=5, type=int, help='beam size for beam search decoding')
    parser.add_argument('--alpha', default=0.7, type=float, help='length normalization hyperparameter for beam search')
    # BLEU computation arguments
    parser.add_argument('--bleu', action='store_true', help='If set, compute BLEU score after translation')
    parser.add_argument('--reference', type=str, help='Path to the reference file (one sentence per line, required if --bleu is set)')
    
    return parser.parse_args()


def candidate_tokenizer_dirs(dicts_dir):
    """Return likely directories containing tokenizer models."""
    if dicts_dir is None:
        return []
    dicts_dir = os.path.normpath(dicts_dir)
    parent = os.path.dirname(dicts_dir)
    return [
        dicts_dir,
        os.path.join(dicts_dir, 'tokenizers'),
        os.path.join(parent, 'tokenizers'),
        parent,
    ]


def infer_tokenizer_path(args, checkpoint_args, side):
    """Find a tokenizer model from --dicts using checkpoint metadata when possible."""
    lang_attr = {'src': 'source_lang', 'tgt': 'target_lang'}[side]
    tokenizer_attr = {'src': 'src_tokenizer', 'tgt': 'tgt_tokenizer'}[side]
    lang = getattr(args, lang_attr, None) or getattr(checkpoint_args, lang_attr, None)
    checkpoint_path = getattr(checkpoint_args, tokenizer_attr, None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path

    if args.dicts is None:
        raise ValueError(
            f"Missing --{tokenizer_attr.replace('_', '-')}. Provide it directly or pass --dicts "
            "pointing to the tokenizer directory."
        )

    wanted_basename = os.path.basename(checkpoint_path) if checkpoint_path else None
    for directory in candidate_tokenizer_dirs(args.dicts):
        if wanted_basename:
            candidate = os.path.join(directory, wanted_basename)
            if os.path.exists(candidate):
                return candidate

    if lang is None:
        raise ValueError(f"Cannot infer {side} tokenizer because the checkpoint has no {side}_lang.")

    matches = []
    for directory in candidate_tokenizer_dirs(args.dicts):
        matches.extend(glob.glob(os.path.join(directory, f'{lang}-bpe-*.model')))

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple {side} tokenizer candidates for language '{lang}': "
            f"{matches}. Please pass --{tokenizer_attr.replace('_', '-')} explicitly."
        )
    raise FileNotFoundError(
        f"Could not find a {side} tokenizer model for language '{lang}' under --dicts {args.dicts}."
    )


def read_prepared_source(args, src_lang):
    """Load a pickled source split created by preprocess.py."""
    data_path = args.data
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, f'{args.prepared_split}.{src_lang}')
    with open(data_path, 'rb') as f:
        return [torch.tensor(tokens, dtype=torch.long) for tokens in pickle.load(f)]


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'), weights_only=False)
    checkpoint_args = state_dict['args']
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)

    if args.input is None and args.data is None:
        raise ValueError("Provide either --input for raw text or --data for a prepared source split.")

    args.src_tokenizer = args.src_tokenizer or infer_tokenizer_path(args, checkpoint_args, 'src')
    args.tgt_tokenizer = args.tgt_tokenizer or infer_tokenizer_path(args, checkpoint_args, 'tgt')

    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)
    # make_batch = utils.make_batch_input(device='cuda' if args.cuda else 'cpu',
    #                                     pad=src_tokenizer.pad_id(),
    #                                     max_seq_len=args.max_len)
    
    


    # batch input sentences
    def batch_iter(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i+batch_size]
    
    # Build model and criterion
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    DEVICE = 'cuda' if args.cuda and torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))

    if args.data is not None:
        src_encoded = read_prepared_source(args, args.source_lang)
    else:
        # Read input sentences
        with open(args.input, encoding="utf-8") as f:
            src_lines = [line.strip() for line in f if line.strip()]

        # Encode input sentences
        src_encoded = [torch.tensor(src_tokenizer.Encode(line, out_type=int, add_eos=True)) for line in src_lines]
    # trim to max_len
    max_seq_len = min(model.encoder.pos_embed.size(1), args.max_len)
    # src_encoded = [s[:max_seq_len] for s in src_encoded]
    src_encoded = [s if len(s)<=max_seq_len else s[:max_seq_len] for s in src_encoded]

    # DEVICE already set above based on hardware availability
    PAD = src_tokenizer.pad_id()
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    print(f'PAD ID: {PAD}, BOS ID: {BOS}, EOS ID: {EOS}\n\
          PAD token: "{src_tokenizer.IdToPiece(PAD)}", BOS token: "{tgt_tokenizer.IdToPiece(BOS)}", EOS token: "{tgt_tokenizer.IdToPiece(EOS)}"')

    

    # Clear output file
    if args.output is not None:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', encoding="utf-8") as out_file:
            out_file.write('')

    def postprocess_ids(ids, pad, bos, eos):
        """Remove leading BOS, truncate at first EOS, remove PADs."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # remove leading BOS if present
        if len(ids) > 0 and ids[0] == bos:
            ids = ids[1:]
        # truncate at EOS (do not include EOS)
        if eos in ids:
            ids = ids[:ids.index(eos)]
        # remove PAD tokens (typically trailing, but remove any)
        ids = [i for i in ids if i != pad]
        return ids

    def decode_sentence(tokenizer: spm.SentencePieceProcessor, sentence_ids):
        """Convert token ids to a detokenized string using the target tokenizer."""
        ids = postprocess_ids(sentence_ids, PAD, BOS, EOS)
        # Use tokenizer.Decode to produce properly detokenized text
        return tokenizer.Decode(ids)
    

    translations = []
    start_time = time.perf_counter()

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            pass

    make_batch = utils.make_batch_input(device=DEVICE, pad=src_tokenizer.pad_id(), max_seq_len=args.max_len)


    #------------------------------------------
    # Translation loop (batched)
    for batch in tqdm(batch_iter(src_encoded, args.batch_size)):
        with torch.no_grad():
            # Pad the batch to the same length
            batch_lengths = [len(x) for x in batch]
            max_len = max(batch_lengths)
            batch_padded = [
                torch.cat([x, torch.full((max_len - len(x),), PAD, dtype=torch.long)]) if len(x) < max_len else x
                for x in batch
            ]
            src_tokens = torch.stack(batch_padded).to(DEVICE)

            # Create a dummy target tensor (all PADs, same shape as src_tokens)
            dummy_y = torch.full_like(src_tokens, fill_value=src_tokenizer.pad_id())

            # Use make_batch to get masks (trg_in, trg_out are not used for inference)
            src_tokens, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch(src_tokens, dummy_y)

            #-----------------------------------------
            # Decode without teacher forcing
            if args.beam_size == 1:
                prediction = decode(model=model,
                                      src_tokens=src_tokens,
                                      src_pad_mask=src_pad_mask,
                                      max_out_len=args.max_len,
                                      tgt_tokenizer=tgt_tokenizer,
                                      args=args,
                                      device=DEVICE)
            else:
                prediction = beam_search_decode(model=model,
                                              src_tokens=src_tokens,
                                              src_pad_mask=src_pad_mask,
                                              max_out_len=args.max_len,
                                              tgt_tokenizer=tgt_tokenizer,
                                              args=args,
                                              device=DEVICE,
                                              beam_size=args.beam_size,
                                              alpha=args.alpha)
            #----------------------------------------

        # Remove BOS and decode each sentence
        for sent in prediction:
            translation = decode_sentence(tgt_tokenizer, sent)
            translations.append(translation)
            if args.output is not None:
                with open(args.output, 'a', encoding="utf-8") as out_file:
                    out_file.write(translation + '\n')
    #------------------------------------------
    logging.info(f'Wrote {len(translations)} lines to {args.output}')
    end_time = time.perf_counter()
    logging.info(f'Translation completed in {end_time - start_time:.2f} seconds')

    # Compute BLEU score if requested
    if getattr(args, 'bleu', False):
        with open(args.reference, encoding='utf-8') as ref_file:
            references = [line.strip() for line in ref_file if line.strip()]
        if len(references) != len(translations):
            raise ValueError(f"Reference ({len(references)}) and hypothesis ({len(translations)}) line counts do not match.")
        bleu = sacrebleu.corpus_bleu(translations, [references])
        print(f"BLEU score: {bleu.score:.2f}")


if __name__ == '__main__':
    args = get_args()
    # make sure --reference is provided if --bleu is set
    if getattr(args, 'bleu', False):
        if not args.reference:
            raise ValueError("You must provide --reference when using --bleu.")
        
    main(args)
