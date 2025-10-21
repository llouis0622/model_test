"""Alert assignment helpers for post-processing predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import THRESHOLDS


def rolling_mean(series: pd.Series, k: int) -> pd.Series:
    return series.rolling(window=k, min_periods=1).mean()


def assign_alert(prob: pd.Series) -> pd.Series:
    thresholds = THRESHOLDS
    t_y, t_o, t_r, delta = (
        thresholds["yellow"],
        thresholds["orange"],
        thresholds["red"],
        thresholds["delta"],
    )
    k = thresholds["persistence_k"]
    pbar = rolling_mean(prob, k)
    labels = np.where(
        (prob >= t_r) & (pbar >= t_r - delta),
        "RED",
        np.where(prob >= t_o, "ORANGE", np.where(prob >= t_y, "YELLOW", "GREEN")),
    )
    return pd.Series(labels, index=prob.index)


def _label_series(s: pd.Series, q_y=0.80, q_o=0.90, q_r=0.97) -> pd.Series:
    y = s.quantile(q_y)
    o = s.quantile(q_o)
    r = s.quantile(q_r)

    def lab(v: float) -> str:
        if v >= r:
            return "RED"
        if v >= o:
            return "ORANGE"
        if v >= y:
            return "YELLOW"
        return "GREEN"

    return s.apply(lab)


def assign_alert_by_quantile(
    df: pd.DataFrame,
    group_cols=None,
    score_col="p_final",
    q_y=0.80,
    q_o=0.90,
    q_r=0.97,
) -> pd.Series:
    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not in DataFrame")

    cols = []
    if group_cols:
        cols = [c for c in group_cols if c in df.columns]

    if not cols:
        fallback_priority = [
            ["HPSN_MCT_ZCD_NM", "HPSN_MCT_BZN_CD_NM", "TA_YM"],
            ["HPSN_MCT_ZCD_NM", "TA_YM"],
            ["HPSN_MCT_BZN_CD_NM", "TA_YM"],
            ["TA_YM"],
            [],
        ]
        for candidate in fallback_priority:
            valid = [c for c in candidate if c in df.columns]
            if len(valid) == len(candidate):
                cols = valid
                break

    if cols:
        return df.groupby(cols, group_keys=False)[score_col].apply(
            lambda s: _label_series(s, q_y, q_o, q_r)
        )
    return _label_series(df[score_col], q_y, q_o, q_r)
