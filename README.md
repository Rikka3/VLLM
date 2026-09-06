# VLLM

Trained a 100M parameter language model where most of the attention layers are replaced with a fixed-size memory instead of the usual growing cache. It scores about the same as a normal transformer of the same size, but uses way less memory when running, and can write and erase documents from its memory at will without retraining from s

## What it does

- Matches a transformer baseline within 0.07 validation loss (ppl 55 vs 52) on the same data and training
- Uses a flat 1.2MB memory state instead of a cache that grows with every token
- Can memorize a document, answer questions about it, then erase it — the knowledge is gone
- Trains on a free Kaggle T4 in about 15 GPU hours

## The demo

```
>>> memorize("The secret code for the vault is BLUE-FALCON-77")
>>> ask("What is the secret code for the vault?")
"BLUE-FALCON-77."

>>> erase()
>>> ask("What is the secret code for the vault?")
"- The first floor of the building is the first..."
```

## Files

- `vsa_lm/` — the model, custom Triton kernels, training code, tests
- `demo.py` — the memorize/erase demo
- `results.md` — all measured numbers from the runs

## Requirements

Python 3.10+, CUDA GPU. `pip install -r requirements.txt`

## Honest limitations

- It ties but doesn't beat the transformer on quality
- Trains about 25% slower at short context
- Only tested up to 512 token context
- The model itself is small (100M params, ~500M tokens of training data — a fraction of what production models see)
