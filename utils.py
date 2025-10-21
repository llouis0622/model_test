"""Utility helpers for preprocessing and risk computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable

from config import BIN2RANK, EPS, VERY_NEGATIVE_SV


def parse_bin_string(values: pd.Series) -> pd.Series:
    """Convert categorical bin labels into numeric ranks in [0, 1]."""
    s = values.astype(str).str.strip()
    normalized = (
        s.str.replace(r"\s*", "", regex=True)
        .str.replace("~", "-", regex=False)
        .str.replace("미만", "이하", regex=False)
        .str.replace("초과", "초과", regex=False)
    )

    ranks = normalized.map(BIN2RANK)

    range_mask = ranks.isna() & normalized.str.contains(r"^\d{1,3}\-?\d{1,3}%$")
    if range_mask.any():
        subset = normalized[range_mask].str.replace("%", "", regex=False)
        lo = subset.str.extract(r"^(\d{1,3})\-?")[0].astype(float)
        hi = subset.str.extract(r"\-(\d{1,3})$")[0].astype(float)
        ranks.loc[range_mask] = ((lo + hi) / 200.0).clip(0.0, 1.0)

    hi_mask = ranks.isna() & normalized.str.contains(r"^\d{1,3}%초과$")
    if hi_mask.any():
        val = normalized[hi_mask].str.replace("%초과", "", regex=False).astype(float) / 100.0
        ranks.loc[hi_mask] = np.minimum(0.95, np.maximum(0.90, val + 0.05))

    lo_mask = ranks.isna() & normalized.str.contains(r"^\d{1,3}%이하$")
    if lo_mask.any():
        val = normalized[lo_mask].str.replace("%이하", "", regex=False).astype(float) / 100.0
        ranks.loc[lo_mask] = np.maximum(0.05, np.minimum(0.10, val - 0.05))

    ranks = ranks.fillna(pd.to_numeric(s, errors="coerce"))
    return ranks.astype(float)


def map_bin_to_rank(values: pd.Series) -> pd.Series:
    """Public helper exposed for preprocessing."""
    return parse_bin_string(values)


def coerce_month_col(df: pd.DataFrame, ym_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[ym_col].astype(str), errors="coerce")
    out[ym_col] = pd.to_datetime(dt.dt.to_period("M").astype(str))
    return out


def as_month_sorted(df: pd.DataFrame, ym_col: str) -> pd.DataFrame:
    return coerce_month_col(df, ym_col).sort_values([ym_col])


def nz(values: Iterable) -> np.ndarray | pd.Series:
    if isinstance(values, pd.Series):
        return values.astype("float64").clip(lower=0.0, upper=1.0)
    arr = np.asarray(values, dtype="float64")
    return np.minimum(1.0, np.maximum(0.0, arr))


def relu_minus(values: Iterable) -> np.ndarray | pd.Series:
    if isinstance(values, pd.Series):
        return (-values).clip(lower=0.0)
    arr = np.asarray(values, dtype="float64")
    return np.maximum(0.0, -arr)


def logistic(values: Iterable) -> np.ndarray | pd.Series:
    arr = np.asarray(values, dtype="float64")
    return 1.0 / (1.0 + np.exp(-arr))


def to_percent(values: Iterable) -> np.ndarray | pd.Series:
    if isinstance(values, pd.Series):
        return (values.astype("float64") / 100.0).clip(lower=0.0, upper=1.0)
    arr = np.asarray(values, dtype="float64") / 100.0
    return np.minimum(1.0, np.maximum(0.0, arr))


def safe_nan(values: pd.Series) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    result[result <= VERY_NEGATIVE_SV] = np.nan
    return result


def group_roll_median(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    return values.rolling(window=window, min_periods=min_periods).median()


def group_roll_mad(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    med = group_roll_median(values, window, min_periods)
    mad = (values - med).abs().rolling(window=window, min_periods=min_periods).median()
    return 1.4826 * mad


def robust_z(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    med = group_roll_median(values, window, min_periods)
    mad = group_roll_mad(values, window, min_periods)
    return (values - med) / (mad + EPS)


def mom(values: pd.Series) -> pd.Series:
    return values - values.shift(1)
