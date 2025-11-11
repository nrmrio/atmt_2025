from typing import List
import torch
import torch.nn.functional as F
from .beam import BeamSearch, BeamSearchNode


def _pad_mask(y: torch.Tensor, pad_id: int):
    return (y == pad_id).unsqueeze(1)  # [B,1,T]

@torch.no_grad()
def decode2(model, src_tokens, src_pad_mask, max_out_len, tgt_tok, args, device) -> List[List[int]]:
    BOS, EOS, PAD = tgt_tok.bos_id(), tgt_tok.eos_id(), tgt_tok.pad_id()
    mode = getattr(args, "decoding", "greedy")
    beam_size = int(getattr(args, "beam_size", 5))
    alpha = float(getattr(args, "alpha", 0.0))
    no_repeat = int(getattr(args, "no_repeat_ngram_size", 0))

    B = src_tokens.size(0)
    out_all: List[List[int]] = []

    if mode == "greedy":
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_out_len):
            logits = model(src_tokens, src_pad_mask, ys, _pad_mask(ys, PAD))
            step = logits[:, -1, :]
            next_ids = step.argmax(dim=-1)

            if no_repeat > 0 and ys.size(1) >= no_repeat:
                for b in range(B):
                    if finished[b]: continue
                    prev = {}
                    yb = ys[b].tolist()
                    for i in range(len(yb) - no_repeat + 1):
                        key = tuple(yb[i:i+no_repeat-1])
                        prev.setdefault(key, set()).add(yb[i+no_repeat-1])
                    prefix = tuple(yb[-(no_repeat-1):]) if no_repeat > 1 else tuple()
                    banned = prev.get(prefix, set())
                    if next_ids[b].item() in banned:
                        order = step[b].argsort(descending=True)
                        for cand in order:
                            if cand.item() not in banned:
                                next_ids[b] = cand
                                break

            ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
            finished |= (next_ids == EOS)
            if finished.all(): break

        for b in range(B):
            seq = ys[b].tolist()
            if EOS in seq: seq = seq[: seq.index(EOS)+1]
            out_all.append(seq)
        return out_all

    # Beam (per-sample)
    for b in range(B):
        s = src_tokens[b:b+1]
        sm = src_pad_mask[b:b+1]
        search = BeamSearch(beam_size, max_out_len, PAD)
        root = BeamSearchNode(torch.tensor([BOS], device=device), 0.0, 1)
        search.add(score=-root.eval(alpha), node=root)

        while True:
            batch = search.get_current_beams()
            if not batch: break

            for score, node in batch:
                seq = node.sequence
                if seq[-1].item() == EOS or node.length >= max_out_len:
                    search.add_final(score, node); continue

                y = seq.unsqueeze(0)
                logits = model(s, sm, y, _pad_mask(y, PAD))
                step = logits[:, -1, :].squeeze(0)
                logp = F.log_softmax(step, dim=-1)

                if no_repeat > 0 and seq.numel() >= no_repeat:
                    prev = {}
                    yb = seq.tolist()
                    for i in range(len(yb) - no_repeat + 1):
                        key = tuple(yb[i:i+no_repeat-1])
                        prev.setdefault(key, set()).add(yb[i+no_repeat-1])
                    prefix = tuple(yb[-(no_repeat-1):]) if no_repeat > 1 else tuple()
                    banned = prev.get(prefix, set())
                    if banned:
                        logp[torch.tensor(list(banned), device=device)] = -1e9

                topk_lp, topk_ix = torch.topk(logp, k=beam_size)
                for lp, wid in zip(topk_lp.tolist(), topk_ix.tolist()):
                    new_seq = torch.cat([seq, torch.tensor([wid], device=device)])
                    new_node = BeamSearchNode(new_seq, node.logp + lp, node.length + 1)
                    score_new = -new_node.eval(alpha)
                    if wid == EOS: search.add_final(score_new, new_node)
                    else:          search.add(score_new, new_node)

            search.prune()
            if search.final.qsize() >= beam_size and search.nodes.qsize() == 0: break

        _, best = search.get_best()
        seq = best.sequence.tolist()
        if EOS in seq: seq = seq[: seq.index(EOS)+1]
        out_all.append(seq)

    return out_all
