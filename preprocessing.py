"""Data loading and normalization utilities used by the pipeline."""
from __future__ import annotations

import pandas as pd

from utils import as_month_sorted, map_bin_to_rank, safe_nan

KEY_MCT = "ENCODED_MCT"
KEY_YM = "TA_YM"


def load_and_join(ds1: pd.DataFrame, ds2: pd.DataFrame, ds3: pd.DataFrame) -> pd.DataFrame:
    """Join source tables into a single modelling DataFrame."""
    merchants = ds1.copy()
    usage = as_month_sorted(ds2.copy(), KEY_YM)
    customers = as_month_sorted(ds3.copy(), KEY_YM)

    for col in usage.columns:
        if col not in (KEY_MCT, KEY_YM):
            usage[col] = safe_nan(usage[col])
    for col in customers.columns:
        if col not in (KEY_MCT, KEY_YM):
            customers[col] = safe_nan(customers[col])

    df = usage.merge(customers, on=[KEY_MCT, KEY_YM], how="left", suffixes=("", "_C"))
    df = df.merge(merchants, on=[KEY_MCT], how="left")
    return df


def normalize_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical bin columns into numeric ranks."""
    df = df.copy()
    bin_cols = [
        "RC_M1_SAA",
        "RC_M1_TO_UE_CT",
        "RC_M1_UE_CUS_CN",
        "RC_M1_AV_NP_AT",
        "APV_CE_RAT",
        "MCT_OPE_MS_CN",
    ]
    for col in bin_cols:
        rank_col = f"{col}_RANK"
        if col in df.columns:
            df[rank_col] = map_bin_to_rank(df[col]).fillna(0.5)
        else:
            df[rank_col] = 0.5
    return df
