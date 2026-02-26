"""Yahoo Answers Topics data loader for AlphaScale — cached version.

The key optimisation: all text is tokenized ONCE at module load time,
stored as pre-tokenized tensors, then sliced per experiment.

This means the expensive parquet read + tokenization happens only once
per Python process regardless of how many scale/fraction combinations
are run. Each subsequent call to load_agnews just slices tensors — 
takes milliseconds instead of minutes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

VAL_SIZE = 10_000  # fixed val set — enough for reliable evaluation

# ── Module-level cache — populated on first call, reused thereafter ───────
_cache: Dict = {}


class SliceDataset(Dataset):
    """Dataset that slices pre-tokenized tensors by index list.

    Avoids duplicating large tensors in memory — shares the underlying
    storage and only materialises the needed rows at __getitem__ time.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        indices: List[int],
    ) -> None:
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = labels
        self.indices        = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        i = self.indices[idx]
        return (
            {
                "input_ids":      self.input_ids[i],
                "attention_mask": self.attention_mask[i],
            },
            int(self.labels[i]),
        )


def _build_cache(data_path: Path, max_seq_len: int, seed: int) -> None:
    """Read parquet, tokenize everything once, store in _cache."""

    print(f"  [Yahoo] Reading parquet files...", flush=True)
    train_df = pd.concat([
        pd.read_parquet(data_path / "train-00000-of-00002.parquet"),
        pd.read_parquet(data_path / "train-00001-of-00002.parquet"),
    ], ignore_index=True)
    test_df = pd.read_parquet(data_path / "test-00000-of-00001.parquet")
    print(f"  [Yahoo] {len(train_df):,} train / {len(test_df):,} test rows", flush=True)

    def make_text(df: pd.DataFrame) -> List[str]:
        t = df["question_title"].fillna("").astype(str)
        c = df["question_content"].fillna("").astype(str)
        a = df["best_answer"].fillna("").astype(str)
        return (t + " " + c + " " + a).tolist()

    train_texts  = make_text(train_df)
    train_labels = train_df["topic"].tolist()
    test_texts   = make_text(test_df)
    test_labels  = test_df["topic"].tolist()

    # ── Fixed val split ───────────────────────────────────────────────────
    rng = np.random.RandomState(seed)
    n_total    = len(train_texts)
    all_idx    = rng.permutation(n_total)
    val_idx    = all_idx[:VAL_SIZE].tolist()
    train_pool = all_idx[VAL_SIZE:].tolist()   # remaining for train fractions

    # ── Tokenize everything ONCE ──────────────────────────────────────────
    bert_path = str(data_path.parent / "bert-base-uncased")
    print(f"  [Yahoo] Loading tokenizer from {bert_path}...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    def tokenize(texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    # Tokenize val texts (10K — fast)
    val_texts  = [train_texts[i] for i in val_idx]
    val_labels_list = [train_labels[i] for i in val_idx]
    print(f"  [Yahoo] Tokenizing {len(val_texts):,} val samples...", flush=True)
    val_ids, val_mask = tokenize(val_texts)

    # Tokenize full train pool (1.26M — slow, but only once)
    pool_texts  = [train_texts[i] for i in train_pool]
    pool_labels = [train_labels[i] for i in train_pool]
    print(f"  [Yahoo] Tokenizing {len(pool_texts):,} train pool samples (once only)...", flush=True)
    pool_ids, pool_mask = tokenize(pool_texts)

    # Tokenize test set
    print(f"  [Yahoo] Tokenizing {len(test_texts):,} test samples...", flush=True)
    test_ids, test_mask = tokenize(test_texts)

    _cache["pool_ids"]      = pool_ids
    _cache["pool_mask"]     = pool_mask
    _cache["pool_labels"]   = torch.tensor(pool_labels, dtype=torch.long)
    _cache["pool_size"]     = len(pool_texts)
    _cache["val_ids"]       = val_ids
    _cache["val_mask"]      = val_mask
    _cache["val_labels"]    = torch.tensor(val_labels_list, dtype=torch.long)
    _cache["test_ids"]      = test_ids
    _cache["test_mask"]     = test_mask
    _cache["test_labels"]   = torch.tensor(test_labels, dtype=torch.long)
    _cache["rng"]           = rng   # keep rng state for consistent sampling
    print(f"  [Yahoo] Cache built. Subsequent calls will reuse tokenized tensors.", flush=True)


def load_agnews(
    data_path: str,
    dataset_fraction: float = 1.0,
    batch_size: int = 128,
    max_seq_len: int = 128,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load Yahoo Answers and return DataLoaders.

    First call: reads parquet + tokenizes everything (~5-10 min).
    Subsequent calls: slices pre-tokenized tensors (milliseconds).

    Named load_agnews for drop-in compatibility with scaling_runner.py.
    """
    data_path = Path(data_path)

    # Build cache on first call only
    if not _cache:
        _build_cache(data_path, max_seq_len, seed)
    else:
        print(f"  [Yahoo] Using cached tokenized tensors.", flush=True)

    # ── Slice train fraction from pool ────────────────────────────────────
    pool_size = _cache["pool_size"]
    n_train   = max(1, int(dataset_fraction * pool_size))
    # Use first n_train indices — deterministic, no re-shuffling needed
    # (pool was already randomly permuted during cache build)
    train_indices = list(range(n_train))
    val_indices   = list(range(len(_cache["val_labels"])))
    test_indices  = list(range(len(_cache["test_labels"])))

    print(f"  [Yahoo] Fraction={dataset_fraction} → "
          f"{n_train:,} train / {len(val_indices):,} val / {len(test_indices):,} test",
          flush=True)

    train_dataset = SliceDataset(
        _cache["pool_ids"], _cache["pool_mask"], _cache["pool_labels"], train_indices
    )
    val_dataset = SliceDataset(
        _cache["val_ids"], _cache["val_mask"], _cache["val_labels"], val_indices
    )
    test_dataset = SliceDataset(
        _cache["test_ids"], _cache["test_mask"], _cache["test_labels"], test_indices
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
