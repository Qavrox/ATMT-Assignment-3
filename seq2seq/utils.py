import os
import logging
import torch
import sys
import sentencepiece as spm

from collections import defaultdict
from torch.serialization import default_restore_location

# this file needs some cleanup at some point, some functions are not used anymore :]

def save_embedding_layer(embedding_layer, file_path):
    """
    Save the weights of an embedding layer to a file using torch.save.

    Args:
        embedding_layer (nn.Embedding): The embedding layer to save.
        file_path (str): Path to the file where the embedding weights will be saved.
    """
    torch.save(embedding_layer.state_dict(), file_path)

def load_embedding(embed_path, tokenizer):
    """Load pretrained embeddings."""

    embedding = {}
    with open(embed_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            token = values[0]
            vector = list(map(float, values[1:]))
            # Convert token to id using sentencepiece tokenizer
            token_id = tokenizer.piece_to_id(token)
            embedding[token_id] = torch.tensor(vector)
    return embedding


def move_to_device(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device)
    elif isinstance(sample, list):
        return [move_to_device(x, device) for x in sample]
    elif isinstance(sample, dict):
        return {key: move_to_device(value, device) for key, value in sample.items()}
    else:
        return sample


def move_to_cuda(sample):
    return move_to_device(sample, 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


def save_checkpoint(args, model, optimizer, epoch, valid_loss, lr_scheduler=None):
    os.makedirs(args.save_dir, exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    prev_best = getattr(save_checkpoint, 'best_loss', float('inf'))
    save_checkpoint.best_loss = min(prev_best, valid_loss)

    state_dict = {
        'epoch': epoch,
        'val_loss': valid_loss,
        'best_loss': save_checkpoint.best_loss,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
    }
    if lr_scheduler is not None:
        state_dict['lr_scheduler'] = lr_scheduler.state_dict()

    if args.epoch_checkpoints and epoch % args.save_interval == 0:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint{}_{:.3f}.pt'.format(epoch, valid_loss)))
    if valid_loss < prev_best:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_best.pt'))
    if last_epoch < epoch:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_last.pt'))


def load_checkpoint(args, model, optimizer, lr_scheduler=None):
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'), weights_only=False)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        if lr_scheduler is not None and 'lr_scheduler' in state_dict:
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        save_checkpoint.best_loss = state_dict.get('best_loss', float('inf'))
        save_checkpoint.last_epoch = state_dict.get('last_epoch', -1)
        logging.info('Loaded checkpoint {}'.format(checkpoint_path))
        return state_dict


def init_logging(args):
    handlers = [logging.StreamHandler()]
    if hasattr(args, 'log_file') and args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode='w'))
    logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    # Assign a unique ID to each module instance, so that incremental state is not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]
    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def load_tokenizer(tokenizer_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer


def make_batch_input(device, pad: int = 3, max_seq_len: int = None):
    def batch_fn(x, y):
        if max_seq_len is not None:
            x = x[:, :max_seq_len]
            y = y[:, :max_seq_len]
        src = x.to(device)
        tgt_in = y[:, :-1].to(device)
        tgt_out = y[:, 1:].contiguous().view(-1).to(device)
        src_pad_mask = (src == pad).view(src.size(0), 1, 1, src.size(-1))
        tgt_pad_mask = (tgt_in == pad).view(tgt_in.size(0), 1, 1, tgt_in.size(-1))
        return src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask
    return batch_fn
