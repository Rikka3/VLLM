"""Parity harness: exactness checks for kernels, gradients, decoder, erasure."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .decode import VSARecurrentDecoder
from .functional import VSAChunkFunction, vsa_parallel_attention, vsa_recurrent_reference
from .model import VSALanguageModel


def test_oracles():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 96, 32, device="cuda") for _ in range(3))
    gamma = torch.tensor([0.9, 0.999], device="cuda")
    y_rec, M = vsa_recurrent_reference(q, k, v, gamma)
    y_par = vsa_parallel_attention(q, k, v, gamma)
    w = gamma.view(1, 2, 1) ** torch.arange(95, -1, -1, device="cuda", dtype=torch.float32)
    M_cf = torch.einsum('bhte,bhtd->bhed', k * w.unsqueeze(-1), v)
    e1 = (y_rec - y_par).abs().max().item()
    e2 = (M - M_cf).abs().max().item()
    print(f"[oracles]  y: {e1:.2e}  M: {e2:.2e}")
    assert e1 < 1e-4 and e2 < 2e-3


def test_kernel_forward():
    torch.manual_seed(0)
    for (B, H, T, D) in [(2, 4, 64, 32), (1, 24, 512, 32), (1, 12, 512, 64)]:
        q, k, v = (torch.randn(B, H, T, D, device="cuda") for _ in range(3))
        gamma = torch.linspace(0.01, 0.9999, H, device="cuda")
        e = (VSAChunkFunction.apply(q, k, v, gamma, 32)
             - vsa_parallel_attention(q, k, v, gamma)).abs().max().item()
        print(f"[forward]  B{B} H{H} T{T} D{D}: {e:.2e}")
        assert e < 3e-3


def test_kernel_backward():
    torch.manual_seed(1)
    for D in (32, 64):
        q0, k0, v0 = (torch.randn(2, 4, 128, D, device="cuda") for _ in range(3))
        g0 = torch.tensor([0.05, 0.5, 0.9, 0.999], device="cuda")
        dy = torch.randn(2, 4, 128, D, device="cuda")
        a = [t.clone().requires_grad_(True) for t in (q0, k0, v0, g0)]
        (vsa_parallel_attention(a[0], a[1], a[2], a[3]) * dy).sum().backward()
        b = [t.clone().requires_grad_(True) for t in (q0, k0, v0, g0)]
        (VSAChunkFunction.apply(b[0], b[1], b[2], b[3], 32) * dy).sum().backward()
        names = ("dq", "dk", "dv", "dgamma")
        errs = {n: ((x.grad - y.grad).abs().max() / x.grad.abs().max().clamp_min(1e-8)).item()
                for n, x, y in zip(names, a, b)}
        print(f"[backward D{D}]  " + "  ".join(f"{n}: {e:.2e}" for n, e in errs.items()))
        assert max(errs.values()) < 1e-3


def test_gamma_learns():
    torch.manual_seed(3)
    q, k, v = (torch.randn(2, 4, 128, 32, device="cuda") for _ in range(3))
    target = vsa_parallel_attention(q, k, v, torch.full((4,), 0.95, device="cuda")).detach()
    z = torch.full((4,), float(torch.logit(torch.tensor(0.3 / 0.9999))),
                   device="cuda", requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = ((VSAChunkFunction.apply(q, k, v, 0.9999 * torch.sigmoid(z), 32)
                 - target) ** 2).mean()
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    gamma = (0.9999 * torch.sigmoid(z)).detach()
    print(f"[gamma-learns]  loss {first:.4f} -> {loss.item():.6f}  "
          f"gamma {gamma.min():.3f}..{gamma.max():.3f}")
    assert loss.item() < 0.05 * first and (gamma - 0.95).abs().max() < 0.03


def test_end_to_end_decode(hybrid_every=0, dim=128):
    torch.manual_seed(2)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    model = VSALanguageModel(vocab_size=2048, dim=dim, num_heads=4, num_layers=2,
                             max_T=128, hybrid_every=hybrid_every).cuda().float().eval()
    dec = VSARecurrentDecoder(model)
    ids = torch.randint(0, 2048, (1, 96), device="cuda")
    with torch.no_grad():
        full = model.lm_head(model(ids))
        x, states, kv = dec.prefill(ids[:, :64])
        err = (model.lm_head(x) - full[:, 63:64, :]).abs().max().item()
        for t in range(64, 96):
            x, states, kv = dec.step(ids[:, t:t+1], t, states, kv)
            err = max(err, (model.lm_head(x) - full[:, t:t+1, :]).abs().max().item())
    torch.backends.cuda.matmul.allow_tf32 = old
    print(f"[decode parity h{hybrid_every} d{dim}]  max logit diff: {err:.2e}")
    assert err < 5e-3


def test_memorize_forget():
    torch.manual_seed(4)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    model = VSALanguageModel(vocab_size=2048, dim=128, num_heads=4, num_layers=2,
                             max_T=128, hybrid_every=0).cuda().float().eval()
    dec = VSARecurrentDecoder(model)
    doc = torch.randint(0, 2048, (1, 40), device="cuda")
    probe = torch.randint(0, 2048, (1, 8), device="cuda")
    states, kv = VSARecurrentDecoder.blank_state(model, 1, "cuda")
    states, kv, pos, traces = dec.memorize(doc, states, kv, 0)
    for t in range(8):
        x, states, kv = dec.step(probe[:, t:t+1], pos, states, kv)
        pos += 1
    pre = [s.clone() for s in states]
    ref = torch.cat([doc, probe, torch.randint(0, 2048, (1, 16), device="cuda")], dim=1)
    err = (model.lm_head(x[:, -1]) - model.lm_head(model(ref))[:, 47, :]).abs().max().item()
    st, kvz = VSARecurrentDecoder.blank_state(model, 1, "cuda")
    st, kvz, pos_z, trz = dec.memorize(doc, st, kvz, 0)
    st = dec.forget(trz, st, next_pos=pos_z)
    e_zero = max(s.abs().max().item() for s in st)
    sb, kvb = VSARecurrentDecoder.blank_state(model, 1, "cuda")
    sb, kvb, _, _ = dec.memorize(doc, sb, kvb, 0)
    states = dec.forget(traces, states, next_pos=pos)
    e_decay = max((states[i] - (pre[i] - (dec._gamma(model.blocks[i].vsa).view(1, -1, 1, 1) ** 8) * sb[i])).abs().max().item()
                  for i in range(len(states)) if states[i] is not None)
    torch.backends.cuda.matmul.allow_tf32 = old
    print(f"[memorize/forget]  decode parity: {err:.2e}  zero-erase: {e_zero:.2e}  "
          f"decay-bookkeeping: {e_decay:.2e}")
    assert err < 5e-3 and e_zero < 1e-4 and e_decay < 1e-4


def run_all():
    """Run the full harness; training must not start unless this passes."""
    print("Running parity harness (~90 s)...")
    test_oracles()
    test_kernel_forward()
    test_kernel_backward()
    test_gamma_learns()
    test_end_to_end_decode(0)
    test_end_to_end_decode(2)
    test_end_to_end_decode(0, dim=256)
    test_memorize_forget()
    print("ALL PARITY TESTS PASSED")


if __name__ == "__main__":
    run_all()
