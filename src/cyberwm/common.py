from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def snake(name: str) -> str:
    """Normalize CIC/Zeek column names without changing their meaning."""
    name = name.strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def shannon_entropy(values: pd.Series) -> float:
    """Entropy of a discrete series. 0 means one value dominates completely."""
    if len(values) == 0:
        return 0.0
    counts = values.value_counts(dropna=True)
    if counts.empty:
        return 0.0
    p = counts.to_numpy(dtype=float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def safe_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)
