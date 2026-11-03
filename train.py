import os
import random
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
import sentencepiece as spm
from collections import OrderedDict
import sacrebleu

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq import models, utils
from seq2seq.decode import decode
from seq2seq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

SEED = random.randint(1, 1_000_000_000)


def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')

    # Add data arguments
    parser.add_argument('--data', default='indomain/preprocessed_data/', help='path to data directory')
    parser.add_argument('--source-lang', default='fr', help='source language')
    parser.add_argument('--target-lang', default='en', help='target language')
    parser.add_argument('--src-tokenizer', help='path to source sentencepiece tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', help='path to target sentencepiece tokenizer', required=True)
    parser.add_argument('--max-tokens', default=None, type=int, help='maximum number of tokens in a batch')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--train-on-tiny', action='store_true', help='train model on a tiny dataset')
    
    # # Add model arguments
    parser.add_argument('--arch', default='transformer', choices=ARCH_MODEL_REGISTRY.keys(), help='model architecture')

    # Add optimization arguments
    parser.add_argument('--max-epoch', default=10000, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--warmup-strategy', default='none', choices=['none', 'linear', 'constant'],
                        help='learning rate warmup strategy (none, linear, constant)')
    parser.add_argument('--warmup-steps', default=500, type=int, help='number of warmup steps for lr scheduler')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')
    parser.add_argument('--max-length', default=300, type=int, help='maximum output sequence length during testing')
    # Add checkpoint arguments
    parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--save-dir', default='checkpoints_asg4', help='path to save checkpoints')
    parser.add_argument('--restore-file', default='checkpoint_last.pt', help='filename to load checkpoint')
    parser.add_argument('--save-interval', type=int, default=1, help='save a checkpoint every N epochs')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--epoch-checkpoints', action='store_true', help='store all epoch checkpoints')
    parser.add_argument('--ignore-checkpoints', action='store_true', help='don\'t load any previous checkpoint')
    # Parse twice as model arguments are not known the first time
    args, _ = parser.parse_known_args()
    model_parser = parser.add_argument_group(argument_default=argparse.SUPPRESS)
    ARCH_MODEL_REGISTRY[args.arch].add_args(model_parser)
    args = parser.parse_args()
    ARCH_CONFIG_REGISTRY[args.arch](args)
    return args



def main(args):
    """ Main training function. Trains the translation model over the course of several epochs, including dynamic
    learning rate adjustment and gradient clipping. """
    logging.info('Commencing training!')
    torch.manual_seed(SEED)

    utils.init_logging(args)

    # Load datasets
    def load_data(split):
        return Seq2SeqDataset(
            src_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.source_lang)),
            tgt_file=os.path.join(args.data, '{:s}.{:s}'.format(split, args.target_lang)),
            src_model=args.src_tokenizer, tgt_model=args.tgt_tokenizer)

    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)

    train_dataset = load_data(split='train') if not args.train_on_tiny else load_data(split='tiny_train')
    valid_dataset = load_data(split='valid') if not args.train_on_tiny else load_data(split='tiny_valid')

    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    logging.info('Built a model with {:d} parameters'.format(sum(p.numel() for p in model.parameters())))
    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_id(), reduction='sum')

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Instantiate optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    def get_lr_scheduler(opt, strategy, warmup_steps):
        if strategy == 'linear':
            def lr_lambda(step):
                current_step = step + 1
                if current_step <= warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return (float(warmup_steps) / float(current_step)) ** 0.5
            return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        elif strategy == 'constant':
            def lr_lambda(step):
                current_step = step + 1
                if current_step <= warmup_steps:
                    return 0.1
                return 1.0
            return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return None

    lr_scheduler = get_lr_scheduler(optimizer, getattr(args, 'warmup_strategy', 'none'), getattr(args, 'warmup_steps', 500))

    state_dict = None
    if not args.ignore_checkpoints:
        # Load last checkpoint if one exists
        state_dict = utils.load_checkpoint(args, model, optimizer, lr_scheduler)
    last_epoch = state_dict['last_epoch'] if state_dict is not None else -1
    
    # Track validation performance for early stopping
    bad_epochs = 0
    best_validate = float('inf')

    make_batch = utils.make_batch_input(device=device, pad=src_tokenizer.pad_id(), max_seq_len=args.max_seq_len)
    
    for epoch in range(last_epoch + 1, args.max_epoch):
        train_loader = \
            torch.utils.data.DataLoader(train_dataset, num_workers=0, collate_fn=train_dataset.collater,
                                        batch_sampler=BatchSampler(train_dataset, args.max_tokens, args.batch_size, 1,
                                                                   0, shuffle=True, seed=SEED))
        model.train()
        stats = OrderedDict()
        stats['loss'] = 0
        stats['lr'] = 0
        stats['num_tokens'] = 0
        stats['batch_size'] = 0
        stats['grad_norm'] = 0
        stats['clip'] = 0
        
        # Display progress
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False,
                            # update progressbar every 2 seconds
                            mininterval=2.0)

        # Iterate over the training set
        start_time = time.perf_counter()
        for i, sample in enumerate(progress_bar):
            if args.cuda:
                sample = utils.move_to_cuda(sample)
            if len(sample) == 0:
                continue
            model.train()
            src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch(x=sample['src_tokens'],
                                                                           y=sample['tgt_tokens'])
            
            output = model(src, src_pad_mask, trg_in, trg_pad_mask).to(device)

            loss = \
                criterion(output.view(-1, output.size(-1)), trg_out) / len(sample['src_lengths'])

            if torch.isnan(loss).any():
                logging.warning('Loss is NAN!')
                print(src_tokenizer.Decode(sample['src_tokens'].tolist()[0]), '---', tgt_tokenizer.Decode(sample['tgt_tokens'].tolist()[0]))
                # print()
                # import pdb;pdb.set_trace()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            # Update statistics for progress bar
            total_loss, num_tokens, batch_size = loss.item(), sample['num_tokens'], len(sample['src_tokens'])

            stats['loss'] += total_loss * len(sample['src_lengths']) / sample['num_tokens']
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += num_tokens / len(sample['src_tokens'])
            stats['batch_size'] += batch_size
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
                                     refresh=False)
        # measure time to complete epoch (training only)
        epoch_time = time.perf_counter()- start_time
        
        logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
            value / len(progress_bar)) for key, value in stats.items())))
        logging.info(f'Time to complete epoch {epoch:03d} (training only): {epoch_time:.2f} seconds')


        # Calculate validation loss
        valid_perplexity = validate(args, model, criterion, valid_dataset, epoch, batch_fn=make_batch, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        model.train()
        
        # Save checkpoints
        if epoch % args.save_interval == 0:
            utils.save_checkpoint(args, model, optimizer, epoch, valid_perplexity, lr_scheduler)

        # Check whether to terminate training
        if valid_perplexity < best_validate:
            best_validate = valid_perplexity
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            logging.info('No validation set improvements observed for {:d} epochs. Early stop!'.format(args.patience))
            break

    # Final evaluation on the test set
    test_dataset = load_data(split='test') if not args.train_on_tiny else load_data(split='tiny_valid')
    logging.info('Loading the best model for final evaluation on the test set')
    utils.load_checkpoint(args, model, optimizer)

    # Evaluate the model on the test set
    bleu_score, all_hypotheses, all_references = evaluate(
        args,
        model,
        test_dataset,
        batch_fn=make_batch,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    logging.info('Final Test Set Results: BLEU {:.2f}'.format(bleu_score))


def validate(args, model, criterion, valid_dataset, epoch,
             batch_fn: callable,
             src_tokenizer: spm.SentencePieceProcessor,
             tgt_tokenizer: spm.SentencePieceProcessor):
    """ Validates model performance on a held-out development set. """
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset, num_workers=0, collate_fn=valid_dataset.collater,
                                    batch_sampler=BatchSampler(valid_dataset, args.max_tokens, args.batch_size, 1, 0,
                                                               shuffle=False, seed=SEED))
    model.eval()
    stats = OrderedDict()
    stats['valid_loss'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    all_references = []  # list of reference strings
    all_hypotheses = []  # list of hypothesis strings

    progress_bar = tqdm(valid_loader, desc='| Validating Epoch {:03d}'.format(epoch), leave=False, disable=False, 
                        # update progressbar every 2 seconds
                        mininterval=2.0)
    # Iterate over the validation set
    for i, sample in enumerate(progress_bar):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue
        with torch.no_grad():
            src_tokens, trg_in, trg_out, src_pad_mask, trg_pad_mask = batch_fn(x=sample['src_tokens'],
                                                                               y=sample['tgt_tokens'])
            # Compute loss (with teacher forcing)
            output = model(src_tokens, src_pad_mask, trg_in, trg_pad_mask).to(device)
            loss = criterion(output.view(-1, output.size(-1)), trg_out.view(-1))
            
            # Decoding for BLEU (no teacher forcing)
            predicted_tokens = decode(model=model,
                                      src_tokens=src_tokens,
                                      src_pad_mask=src_pad_mask,
                                      max_out_len=args.max_length,
                                      tgt_tokenizer=tgt_tokenizer,
                                      args=args,
                                      device=device)

        # Update tracked statistics
        stats['valid_loss'] += loss.item()
        stats['num_tokens'] += sample['num_tokens']
        stats['batch_size'] += len(sample['src_tokens'])

        # Collect references and hypotheses for BLEU
        if tgt_tokenizer is not None:
            for ref_tgt, hyp_src in zip(sample['tgt_tokens'], predicted_tokens):
                ref_sentence = tgt_tokenizer.Decode(ref_tgt.tolist())
                hyp_sentence = tgt_tokenizer.Decode(hyp_src)

                all_references.append(ref_sentence)
                all_hypotheses.append(hyp_sentence)

    # Calculate validation perplexity
    stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens']
    perplexity = np.exp(stats['valid_loss'])
    stats['num_tokens'] = stats['num_tokens'] / stats['batch_size']

    # Compute BLEU with sacrebleu
    bleu_score = None
    if src_tokenizer is not None and tgt_tokenizer is not None and len(all_hypotheses) > 0:
        bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
        bleu_score = bleu.score

    # Logging
    logging.info(
        'Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity) +
        ('' if bleu_score is None else ' | BLEU {:.3f}'.format(bleu_score))
    )

    return perplexity
    

def evaluate(args, model, test_dataset,
    batch_fn: callable,
    src_tokenizer: spm.SentencePieceProcessor,
    tgt_tokenizer: spm.SentencePieceProcessor,
    decode_kwargs: dict = None,
):
    """Evaluates the model on a test set using sacrebleu.
       decode_fn: function that generates translations.
       decode_kwargs: dict of extra parameters for decode_fn.
    """
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=0,
        collate_fn=test_dataset.collater,
        # batch_size != 1 may mess things up with decoding
        batch_sampler=BatchSampler(test_dataset, args.max_tokens, batch_size=1, 
                                   num_shards=1, shard_id=0, shuffle=False, seed=SEED),
    )

    model.eval()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    all_references = []
    all_hypotheses = []
    decode_kwargs = decode_kwargs or {}

    progress_bar = tqdm(test_loader, desc='| Evaluating', leave=False, disable=False,
                        # update progressbar every 2 seconds
                        mininterval=2.0)   
    # Iterate over test set
    for i, sample in enumerate(progress_bar):
        if args.cuda:
            sample = utils.move_to_cuda(sample)
        if len(sample) == 0:
            continue

        with torch.no_grad():
            src_tokens, tgt_in, tgt_out, src_pad_mask, _ = batch_fn(
                x=sample["src_tokens"], y=sample["tgt_tokens"]
            )

            #-----------------------------------------
            # Decode without teacher forcing
            prediction = decode(model=model,
                                      src_tokens=src_tokens,
                                      src_pad_mask=src_pad_mask,
                                      max_out_len=args.max_length,
                                      tgt_tokenizer=tgt_tokenizer,
                                      args=args,
                                      device=device)
            #----------------------------------------

        # Collect hypotheses and references
        for ref, hyp in zip(sample["tgt_tokens"], prediction):
        # the for-loop is technically redundant since batch_size=1, but kept for clarity
            ref_sentence = tgt_tokenizer.Decode(ref.tolist())
            hyp_sentence = tgt_tokenizer.Decode(hyp)

            all_references.append(ref_sentence)
            all_hypotheses.append(hyp_sentence)

    # Compute BLEU with sacrebleu
    bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references])
    bleu_score = bleu.score

    logging.info("Test set results: BLEU {:.3f}".format(bleu_score))

    return bleu_score, all_hypotheses, all_references



if __name__ == '__main__':
    args = get_args()
    args.seed = SEED
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
                            format='%(levelname)s: %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main(args)
