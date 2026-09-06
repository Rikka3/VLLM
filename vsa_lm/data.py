"""Token streaming, caching, and deterministic data loading."""
from __future__ import annotations

import glob
import itertools
import os

import torch
from torch.utils.data import DataLoader, Dataset

TOKEN_CACHE_NAME = "fineweb_edu_500m.pt"


def find_input(filename: str, working_dir: str = "/kaggle/working"):
    """Locate a file in the working dir or any attached input, at any depth."""
    for pattern in (os.path.join(working_dir, filename),
                    f"/kaggle/input/**/{filename}"):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    return None


def load_tokens(target: int = 500_000_000, cache_name: str = TOKEN_CACHE_NAME):
    """Load the token cache, or stream and tokenize FineWeb-Edu up to `target`."""
    import tiktoken
    from tqdm import tqdm

    cache = find_input(cache_name)
    if cache:
        print(f"Loading cached tokens: {cache}")
        return torch.load(cache, weights_only=True)

    print("Downloading and tokenizing FineWeb-Edu (first run only)...")
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      streaming=True, split="train")
    all_tokens, n = [], 0
    pbar = tqdm(ds, desc="Tokenizing", mininterval=5.0)
    for ex in pbar:
        toks = enc.encode_ordinary(ex["text"])
        toks.append(enc.eot_token)
        all_tokens.extend(toks)
        n += len(toks)
        if n % 1_000_000 < 50_000:
            pbar.set_postfix({"tokens": f"{n/1e6:.2f}M"})
        if n >= target:
            break
    tokens = torch.tensor(all_tokens, dtype=torch.int32)
    del all_tokens
    torch.save(tokens, os.path.join("/kaggle/working", cache_name))
    return tokens


class TextDataset(Dataset):
    """Contiguous non-overlapping (input, target) windows of max_len tokens."""

    def __init__(self, tokens, max_len: int):
        self.tokens, self.max_len = tokens, max_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.max_len

    def __getitem__(self, idx):
        s = idx * self.max_len
        chunk = self.tokens[s:s + self.max_len + 1]
        return chunk[:-1].long(), chunk[1:].long()


def build_loader(tokens, batch_size: int, max_len: int, start_step: int = 0,
                 seed: int = 1234, val_tokens: int = 2_000_000):
    """Seeded, deterministic loader. Resumes skip exactly `start_step` batches,
    so every batch is seen at most once across restarts."""
    dataset = TextDataset(tokens[:-val_tokens], max_len)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g,
                        num_workers=2, pin_memory=True, persistent_workers=True)
    it = iter(loader)
    if start_step > 0:
        it = itertools.islice(it, start_step, None)
    return it, len(dataset) // batch_size
