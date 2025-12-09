import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel


def decode(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    args,
    device: torch.device,
):
    """Decodes a sequence without teacher forcing. Uses the model's own predictions."""
    model.eval()
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # Ensure inputs are on the right device
    src_tokens = src_tokens.to(device)
    src_pad_mask = src_pad_mask.to(device)

    # Precompute encoder outputs once (expensive part)
    with torch.no_grad():
        enc_out = model.encoder(src_tokens, src_pad_mask)
        max_len = model.decoder.pos_embed.size(1)

    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_out_len):
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]

        # trg_pad_mask: (batch_size, 1, 1, tgt_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)

        # Only run the decoder, reuse enc_out
        with torch.no_grad():
            output = model.decoder(enc_out, src_pad_mask, generated, trg_pad_mask)

        # Get logits for the last time step
        next_token_logits = output[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token
        generated = torch.cat([generated, next_tokens], dim=1)

        # Check EOS
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break

    # Remove initial BOS and cut off after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[: idx + 1]
        predicted_tokens.append(seq)
    return predicted_tokens


def beam_search_decode(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    args,
    device: torch.device,
    beam_size: int = 5,
    alpha: float = 0.7,
):
    """Beam Search decoding compatible with Transformer-based Seq2Seq models."""
    model.eval()
    BOS, EOS, PAD = (
        tgt_tokenizer.bos_id(),
        tgt_tokenizer.eos_id(),
        tgt_tokenizer.pad_id(),
    )

    # Assume batch size = 1 for beam search (as in original implementation)
    src_tokens = src_tokens.to(device)
    src_pad_mask = src_pad_mask.to(device)

    # Precompute encoder outputs once (expensive part)
    with torch.no_grad():
        enc_out = model.encoder(src_tokens, src_pad_mask)
        max_len = model.decoder.pos_embed.size(1)

    # Each beam entry is (seq_tensor, cumulative_log_score)
    beams = [(torch.tensor([[BOS]], device=device, dtype=torch.long), 0.0)]

    for _ in range(max_out_len):
        new_beams = []

        for seq, score in beams:
            # If this hypothesis already ended with EOS, keep it as-is
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            # Truncate if sequence is longer than decoder's max length
            if seq.size(1) > max_len:
                seq = seq[:, :max_len]

            # trg_pad_mask: (1, 1, 1, tgt_len)
            trg_pad_mask = (seq == PAD)[:, None, None, :]

            with torch.no_grad():
                # Only call decoder, reuse enc_out
                logits = model.decoder(enc_out, src_pad_mask, seq, trg_pad_mask)[:, -1, :]

                # Log-probs and top-k candidates
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # topk_ids[:, k] has shape (1,), unsqueeze to (1, 1) to append along time
                next_token = topk_ids[:, k].unsqueeze(0)  # (1, 1)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # Keep top beam_size hypotheses
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # Stop early if all hypotheses have finished
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = beams[0]
    # Return in the same format as greedy decoding (list of token-id lists)
    return [best_seq.squeeze(0).tolist()]
