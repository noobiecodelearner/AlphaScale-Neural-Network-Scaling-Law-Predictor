"""DBpedia-14 data loader for AlphaScale.

Reads train.parquet / test.parquet produced by fancyzhx/dbpedia_14.
Tokenization happens AFTER fraction sampling so small-fraction
experiments don't pay the cost of tokenizing the full 560K corpus.
Output format: (input_ids, attention_mask) dicts — identical to the
previous AG News loader, so nothing else in the pipeline changes.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class DBpediaDataset(Dataset):
    """DBpedia-14 dataset with BERT tokenization.

    Args:
        encodings: Dict of tokenizer outputs (input_ids, attention_mask).
        labels: List of integer class labels (0-indexed, 0-13).
    """

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item, self.labels[idx]


def load_agnews(
    data_path: str,
    dataset_fraction: float = 1.0,
    batch_size: int = 128,
    max_seq_len: int = 128,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load DBpedia-14 from parquet files and return train/val/test DataLoaders.

    Named load_agnews so scaling_runner.py requires zero changes.
    Tokenization is performed AFTER fraction sampling.

    Args:
        data_path: Directory containing train.parquet and test.parquet.
        dataset_fraction: Fraction of training data to use.
        batch_size: DataLoader batch size.
        max_seq_len: Maximum token sequence length.
        seed: Random seed for subset sampling.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_path = Path(data_path)

    # ── Load parquet (fast columnar read) ─────────────────────────────────
    print(f"  [DBpedia] Reading parquet files...", flush=True)
    train_df = pd.read_parquet(data_path / "train.parquet")
    test_df  = pd.read_parquet(data_path / "test.parquet")
    print(f"  [DBpedia] {len(train_df):,} train / {len(test_df):,} test rows", flush=True)

    # labels are already 0-indexed (0-13)
    train_texts  = (train_df["title"] + " " + train_df["content"]).tolist()
    train_labels = train_df["label"].tolist()
    test_texts   = (test_df["title"]  + " " + test_df["content"]).tolist()
    test_labels  = test_df["label"].tolist()

    # ── Val split: 10% of full training set ───────────────────────────────
    rng = np.random.RandomState(seed)
    n_total = len(train_texts)
    all_idx = rng.permutation(n_total)
    n_val          = max(1, int(0.1 * n_total))
    val_idx        = all_idx[:n_val].tolist()
    train_pool_idx = all_idx[n_val:]

    # ── Apply fraction BEFORE tokenization ────────────────────────────────
    n_train            = max(1, int(dataset_fraction * len(train_pool_idx)))
    selected_train_idx = train_pool_idx[:n_train].tolist()

    train_texts_sel  = [train_texts[i]  for i in selected_train_idx]
    train_labels_sel = [train_labels[i] for i in selected_train_idx]
    val_texts_sel    = [train_texts[i]  for i in val_idx]
    val_labels_sel   = [train_labels[i] for i in val_idx]

    print(f"  [DBpedia] Split → {len(train_texts_sel):,} train / "
          f"{len(val_texts_sel):,} val / {len(test_texts):,} test", flush=True)

    # ── Tokenizer from local BERT folder ──────────────────────────────────
    bert_path = str(data_path.parent / "bert-base-uncased")
    print(f"  [DBpedia] Loading tokenizer from {bert_path}...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    def tokenize(texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

    print(f"  [DBpedia] Tokenizing...", flush=True)
    train_enc = tokenize(train_texts_sel)
    val_enc   = tokenize(val_texts_sel)
    test_enc  = tokenize(test_texts)

    train_dataset = DBpediaDataset(train_enc, train_labels_sel)
    val_dataset   = DBpediaDataset(val_enc,   val_labels_sel)
    test_dataset  = DBpediaDataset(test_enc,  test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader