from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _set_csv_field_size_limit() -> None:
    # The dataset contains some very large fields; pandas' python engine uses csv module.
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except Exception:
        # Best-effort; if it fails we still try to read.
        pass


def parse_first_float(value: object) -> float | np.nan:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return np.nan

    match = _NUM_RE.search(text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except Exception:
        return np.nan


@dataclass(frozen=True)
class DatasetSchema:
    target: str
    drop_columns: Tuple[str, ...]


DEFAULT_SCHEMA = DatasetSchema(target="Price", drop_columns=("Unnamed: 0",))


def load_dataset(path: str | Path, *, schema: DatasetSchema = DEFAULT_SCHEMA) -> pd.DataFrame:
    """Loads the dataset.

    Note: although the file extension is .csv, it is actually tab-separated.
    """

    _set_csv_field_size_limit()
    path = Path(path)

    df = pd.read_csv(path, sep="\t", engine="python")

    # Normalize column names.
    df.columns = [str(c).strip() for c in df.columns]

    for col in schema.drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Strip whitespace from categorical columns.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin({"", "nan", "None", "NULL", "null"}), col] = np.nan

    # Robustly coerce Land size into numeric (some rows contain extremely long malformed strings).
    if "Land size" in df.columns:
        df["Land size"] = df["Land size"].map(parse_first_float).astype(float)

    # Coerce House size to numeric if needed.
    if "House size" in df.columns and df["House size"].dtype == object:
        df["House size"] = df["House size"].map(parse_first_float).astype(float)

    # Ensure target numeric.
    if schema.target in df.columns:
        df[schema.target] = pd.to_numeric(df[schema.target], errors="coerce")

    # Drop rows with missing target.
    df = df.dropna(subset=[schema.target]).reset_index(drop=True)

    return df


def infer_feature_types(df: pd.DataFrame, *, target: str) -> tuple[list[str], list[str], list[str]]:
    feature_columns = [c for c in df.columns if c != target]
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []

    for col in feature_columns:
        if df[col].dtype == object:
            categorical_columns.append(col)
        else:
            numeric_columns.append(col)

    return feature_columns, categorical_columns, numeric_columns


def top_categories(df: pd.DataFrame, categorical_columns: List[str], *, max_values: int = 200) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for col in categorical_columns:
        # keep most frequent values to keep UI usable; still allow free text if needed.
        vc = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .head(max_values)
            .index.tolist()
        )
        out[col] = vc
    return out
