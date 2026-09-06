"""Demo: memorize, ask, erase. Requires a trained checkpoint (vsa_latest.pth).

Run by:  python demo.py
"""

import torch
import tiktoken

from vsa_lm.model import VSALanguageModel
from vsa_lm.decode import VSARecurrentDecoder

enc = tiktoken.get_encoding("gpt2")

model = VSALanguageModel(hybrid_every=4).cuda().eval()
ckpt = torch.load("vsa_latest.pth", map_location="cuda", weights_only=True)
model.load_state_dict(ckpt["model"])
print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

dec = VSARecurrentDecoder(model)

doc = "The secret code for the vault is BLUE-FALCON-77. Only the night manager knows it."
question = "The secret code for the vault is"

doc_ids = torch.tensor(enc.encode_ordinary(doc), device="cuda").view(1, -1)
q_ids = torch.tensor(enc.encode_ordinary(question), device="cuda").view(1, -1)

def snapshot(states, kv):
    st = [s.clone() if s is not None else None for s in states]
    kvc = [None if t is None else (t[0].clone(), t[1].clone()) for t in kv]
    return st, kvc

def ask(question_ids, states, kv, pos, n=12):
    st, kvc = snapshot(states, kv)
    out, _, _, _ = dec.continue_generate(question_ids, st, kvc, pos, n, temperature=0.0)
    return enc.decode(out[0].tolist()).strip()

# --- write the document into memory ---
states, kv = VSARecurrentDecoder.blank_state(model, 1, "cuda")
states, kv, pos, trace = dec.memorize(doc_ids, states, kv, 0)
state_kb = sum(s.numel() for s in states if s is not None) * 4 / 1024
print(f"\nMemorized {doc_ids.shape[1]} tokens. Memory state: {state_kb:.0f} KB")

# --- ask about it ---
print(f"\nQ: {question}")
print(f"A: {ask(q_ids, states, kv, pos)}")

# --- erase it ---(like a bop it toy)
states = dec.forget(trace, states, next_pos=pos)
print(f"\nAfter erasing the document:")
print(f"Q: {question}")
print(f"A: {ask(q_ids, states, kv, pos)}")

# --- check the state is actually clean ---
clean = all(s.abs().max() < 1e-4 for s in states if s is not None)
print(f"\nState returned to zero: {clean}")
