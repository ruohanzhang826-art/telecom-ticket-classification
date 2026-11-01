"""Data loading, text cleaning and train/test splits."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "tickets.csv"

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s'+-]")


def clean_text(text: str) -> str:
    """Lower-case, strip punctuation noise and collapse whitespace."""
    text = text.lower()
    text = _punct.sub(" ", text)
    return _ws.sub(" ", text).strip()


def load_tickets(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["description"].astype(str).map(clean_text)
    return df


def random_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Standard stratified split. Train and test share phrasing patterns."""
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["department"])


def unseen_phrasing_split(df: pd.DataFrame, n_holdout: int = 3, seed: int = 42):
    """Harder split: for every department, hold out a few phrasing templates
    entirely. The test set then contains ways of describing a problem that the
    model has never seen - closer to what happens when a new product or a new
    agent writing style appears in production.
    """
    rng = np.random.default_rng(seed)
    held = []
    for dept, group in df.groupby(df["template_id"].str.rsplit("_", n=1).str[0]):
        templates = sorted(group["template_id"].unique())
        held.extend(rng.choice(templates, size=n_holdout, replace=False))
    mask = df["template_id"].isin(held)
    return df[~mask], df[mask]


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Train on older tickets, test on the most recent ones."""
    df = df.sort_values("created_at")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]
