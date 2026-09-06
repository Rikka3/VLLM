"""Inference-cost benchmarks: decode cost, state size, analytic complexity."""
from __future__ import annotations

import torch

from .decode import VSARecurrentDecoder
from .model import VSALanguageModel


def decode_flops_per_token(model):
    """(slope, flat) MACs per decoded token at batch 1."""
    slope, flat = 0, 0
    for blk in model.blocks:
        if hasattr(blk, "vsa"):
            flat += 2 * blk.vsa.H * blk.vsa.HD * blk.vsa.HD
        else:
            slope += 2 * 2 * blk.softmax_attn.H * blk.softmax_attn.HD
    return slope, flat


def complexity_report(hybrid_model=None):
    """Inference-cost comparison across three architectures."""
    torch.manual_seed(0)
    models = {
        "transformer (8 softmax)": VSALanguageModel(hybrid_every=1).cuda().eval(),
        "hybrid 6+2 (ours)": (hybrid_model if hybrid_model is not None
                              else VSALanguageModel(hybrid_every=4)).cuda().eval(),
        "pure VSA (8 memory)": VSALanguageModel(hybrid_every=0).cuda().eval(),
    }

    print("=== 1. Decode compute per token (analytic MACs, batch 1) ===")
    for name, m in models.items():
        slope, flat = decode_flops_per_token(m)
        print(f"  {name:<26} {slope:>7,} * T + {flat:>9,}")

    print("\n=== 2. Inference state size ===")
    lengths = (512, 8192, 32768)
    print(f"{'model':<26}{'flat':>10}{'per tok':>10}" + "".join(f"{f'@{T:,}':>12}" for T in lengths))
    for name, m in models.items():
        flat, per_tok = VSARecurrentDecoder.state_bytes(m)
        print(f"{name:<26}{flat/1e6:>9.2f}M{per_tok:>9,}" +
              "".join(f"{(flat + per_tok*T)/1e6:>11.2f}M" for T in lengths))
