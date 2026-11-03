import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """
    Decodes a sequence without teacher forcing using greedy decoding.
    Generates tokens autoregressively by taking the argmax token at each time step.
    """
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Pre-encode source tokens once to avoid re-encoding across generation steps
    encoder_out = model.encoder(src_tokens, src_pad_mask)
    
    for t in range(max_out_len):
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        output = model.decoder(encoder_out, src_pad_mask, generated, trg_pad_mask).to(device)
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        generated = torch.cat([generated, next_tokens], dim=1)
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break

    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens


def length_penalty(length: int, alpha: float) -> float:
    """
    Computes length normalization penalty based on the GNMT heuristic (Wu et al., 2016):
    lp(Y) = ((5 + |Y|) / (5 + 1)) ** alpha
    When alpha == 0, lp(Y) = 1.0 (no length penalty).
    """
    if alpha > 0.0:
        return ((5.0 + float(length)) / 6.0) ** alpha
    return 1.0


def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    """
    Beam Search decoding compatible with Transformer-based Seq2Seq models.
    Incorporates length normalization (alpha penalty), encoder caching, and completed hypothesis tracking.
    """
    model.eval()
    if beam_size <= 1:
        return decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, args, device)

    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()

    # Pre-encode source tokens once
    encoder_out = model.encoder(src_tokens, src_pad_mask)

    # __QUESTION 1: what does this line set up and why is the beam represented this way?
    # ANSWER:
    # This initializes the search tree with a single seed hypothesis starting with [BOS].
    # Each beam item is stored as a tuple (hypothesis_tensor, cumulative_log_prob), where the tensor
    # has shape (1, current_len) and cumulative log prob starts at log(1.0) = 0.0.
    # Representing each beam as a (sequence, score) pair allows us to maintain the generated history
    # and sort/prune candidates efficiently at each expansion step.
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    completed = []

    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            # If sequence already hit EOS, do not expand it further
            if seq[0, -1].item() == EOS:
                continue

            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                seq_in = seq[:, :max_len] if seq.size(1) > max_len else seq

                # __QUESTION 2: Why do we need to create trg_pad_mask here and how does it affect the model's predictions?
                # ANSWER:
                # The decoder self-attention requires a mask to prevent attention weights from attending to PAD tokens.
                # Here trg_pad_mask creates a boolean tensor of shape (1, 1, 1, seq_len) where PAD positions are True.
                # In forward computation, positions with True are masked with -inf before softmax, ensuring the model
                # conditions strictly on valid generated tokens without PAD corruption.
                trg_pad_mask = (seq_in == PAD)[:, None, None, :]
                logits = model.decoder(encoder_out, src_pad_mask, seq_in, trg_pad_mask)[:, -1, :]

                # __QUESTION 3: Explain the purpose of applying log_softmax and selecting top-k tokens here.
                # ANSWER:
                # Logits are unnormalized projection scores. log_softmax converts them into proper log probabilities
                # log P(y_t | y_<t, X) that sum to 1 in probability space. Calling topk(beam_size) selects the k most
                # probable token candidates and their log-probs. Pruning to the top-k vocabulary items at each step
                # reduces computational complexity from O(|V|) to O(K) while retaining the most promising search paths.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # __QUESTION 4: explain the tensor shapes and the logic when creating new_seq and new_score below. Is any broadcasting or indexing issue possible?
                # ANSWER:
                # topk_ids has shape (1, beam_size), so topk_ids[:, k] has shape (1,). Calling unsqueeze(0) produces (1, 1).
                # Concatenating with seq of shape (1, t) along dim=1 yields new_seq of shape (1, t+1).
                # For score, topk_log_probs[:, k].item() extracts the scalar log-prob, which is added to the previous
                # scalar cumulative score: score + log P(y_t | y_<t).
                # Notice: .item() assumes batch_size=1; for batch_size > 1, batch-level indexing must be used.
                token_tensor = topk_ids[0, k].unsqueeze(0).unsqueeze(0)  # (1, 1)
                new_seq = torch.cat([seq, token_tensor], dim=1)
                new_score = score + topk_log_probs[0, k].item()

                if token_tensor.item() == EOS:
                    lp = length_penalty(new_seq.size(1) - 1, alpha)
                    completed.append((new_seq, new_score / lp))
                else:
                    new_beams.append((new_seq, new_score))

        if not new_beams:
            break

        # Prune down to top beam_size active hypotheses
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # __QUESTION 5: Why do we check for EOS here and what does it imply for beam search?
        # ANSWER:
        # Checking for EOS identifies completed sentences. In MT, hypotheses finish at variable lengths.
        # Once a sequence generates EOS, continuing generation would append invalid trailing tokens.
        # By routing finished sequences to a completed pool and checking stopping criteria against length-normalized
        # scores, beam search can safely stop early when the top completed hypothesis cannot be surpassed by any open beam.
        if len(completed) >= beam_size:
            best_comp_score = max(c[1] for c in completed)
            best_open_norm_score = max(b[1] / length_penalty(b[0].size(1) - 1, alpha) for b in beams)
            if best_comp_score >= best_open_norm_score:
                break

    # Select best hypothesis: prefer completed hypotheses with best length-normalized score
    if completed:
        best_seq, _ = max(completed, key=lambda x: x[1])
    elif beams:
        best_seq, _ = max(beams, key=lambda x: x[1] / length_penalty(x[0].size(1) - 1, alpha))
    else:
        best_seq = torch.tensor([[BOS, EOS]], device=device)

    # __QUESTION 6: What is returned, and why are we squeezing, converting to list and wrapping in another list here?
    # ANSWER:
    # best_seq has shape (1, seq_len). .squeeze(0) removes the batch dimension -> shape (seq_len,).
    # .tolist() converts the tensor into a Python list of integers.
    # Wrapping it in [...] returns a list of candidate token lists (length 1 for batch_size 1),
    # matching the batched output interface expected by translate.py for subsequent detokenization.
    # Strip BOS token to align with decode() interface
    seq_list = best_seq.squeeze(0).tolist()
    if seq_list and seq_list[0] == BOS:
        seq_list = seq_list[1:]
    if EOS in seq_list:
        eos_idx = seq_list.index(EOS)
        seq_list = seq_list[:eos_idx + 1]
    return [seq_list]
