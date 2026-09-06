"""Held-out evaluation via the training compute path (positions 0..max_T-1)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_ppl(model, tokens, start: int, length: int, chunk: int = 512, bs: int = 4):
    """Perplexity on held-out text using the parallel forward pass.

    The recurrent decoder is invalid past max_T (position labels wrap), so
    all reported numbers must come from this path.
    """
    n_chunks = (length - 1) // chunk
    nll, cnt = 0.0, 0
    for i in range(0, n_chunks, bs):
        j = min(i + bs, n_chunks)
        x = torch.stack([tokens[start + c*chunk: start + c*chunk + chunk] for c in range(i, j)]).long().cuda()
        y = torch.stack([tokens[start + c*chunk + 1: start + c*chunk + chunk + 1] for c in range(i, j)]).long().cuda()
        with torch.autocast("cuda", torch.float16):
            h = model(x)
        logits = model.lm_head(h).float()
        nll += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                               reduction="sum").item()
        cnt += y.numel()
    return math.exp(nll / cnt)


@torch.no_grad()
def five_spot_report(model, tokens, offsets=(400_000, 800_000, 1_200_000, 1_600_000, 1_950_000)):
    """Average held-out loss over five fixed text slices (the fair score)."""
    n = len(tokens)
    losses = []
    for off in offsets:
        ppl = evaluate_ppl(model, tokens, n - off, 4096)
        losses.append(math.log(ppl))
        print(f"  spot {off:>8,} tokens into val region: ppl {ppl:7.2f}  loss {math.log(ppl):.3f}")
    avg = sum(losses) / len(losses)
    print(f"5-SPOT AVERAGE: val_loss {avg:.3f}  val_ppl {math.exp(avg):.2f}")
    return avg
