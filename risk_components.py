"""Rule-based risk component calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EPS, MIN_PERIODS, ROLL_WINDOW
from utils import logistic, mom, nz, relu_minus, robust_z, to_percent


def _zero(series: pd.Series) -> pd.Series:
    return series.fillna(0.0).clip(0.0, 1.0)


def compute_sales_risk(df: pd.DataFrame, key=("ENCODED_MCT", "TA_YM")) -> pd.DataFrame:
    data = df.sort_values(list(key)).copy()
    r_sales = data["RC_M1_SAA_RANK"]
    r_cnt = data["RC_M1_TO_UE_CT_RANK"]
    r_aov = data["RC_M1_AV_NP_AT_RANK"]
    r_cancel = data["APV_CE_RAT_RANK"]

    s_drop = _zero(nz((relu_minus(mom(r_sales)) + relu_minus(mom(r_cnt))) / 2.0))
    s_aov = _zero(nz(relu_minus(robust_z(r_aov, ROLL_WINDOW, MIN_PERIODS))))
    s_cancel = _zero(nz(np.maximum(r_cancel, logistic(robust_z(r_cancel, ROLL_WINDOW, MIN_PERIODS)))))

    ind_rank = _zero(nz(1.0 - to_percent(data.get("M12_SME_RY_SAA_PCE_RT", pd.Series(50.0, index=data.index)))))
    biz_rank = _zero(nz(1.0 - to_percent(data.get("M12_SME_BZN_SAA_PCE_RT", pd.Series(50.0, index=data.index)))))
    s_peer = _zero(nz((ind_rank + biz_rank) / 2.0))

    dlv_raw = data.get("DLV_SAA_RAT", pd.Series(0.0, index=data.index))
    dlv = _zero(nz(to_percent(dlv_raw)))
    s_dlv_jump = _zero(nz(relu_minus(-mom(dlv)) + relu_minus(-robust_z(dlv, ROLL_WINDOW, MIN_PERIODS))))

    sales_risk = _zero(0.35 * s_drop + 0.15 * s_aov + 0.20 * s_cancel + 0.20 * s_peer + 0.10 * (dlv * s_dlv_jump))
    out = data[["ENCODED_MCT", "TA_YM"]].copy()
    out["Sales_Risk"] = sales_risk
    return out


def compute_customer_risk(df: pd.DataFrame, key=("ENCODED_MCT", "TA_YM")) -> pd.DataFrame:
    data = df.sort_values(list(key)).copy()
    r_customers = data["RC_M1_UE_CUS_CN_RANK"]
    s_drop = _zero(nz(relu_minus(mom(r_customers)) + relu_minus(robust_z(r_customers, ROLL_WINDOW, MIN_PERIODS))))

    q_reuse = _zero(nz(to_percent(data.get("MCT_UE_CLN_REU_RAT", pd.Series(0.0, index=data.index)))))
    q_new = _zero(nz(to_percent(data.get("MCT_UE_CLN_NEW_RAT", pd.Series(0.0, index=data.index)))))
    s_loyal = _zero(nz(relu_minus(mom(q_reuse))))
    s_acq = _zero(nz(relu_minus(mom(q_new))))

    age_cols = [
        "M12_MAL_1020_RAT",
        "M12_MAL_30_RAT",
        "M12_MAL_40_RAT",
        "M12_MAL_50_RAT",
        "M12_MAL_60_RAT",
        "M12_FME_1020_RAT",
        "M12_FME_30_RAT",
        "M12_FME_40_RAT",
        "M12_FME_50_RAT",
        "M12_FME_60_RAT",
    ]
    for col in age_cols:
        if col not in data.columns:
            data[col] = 0.0
    weights_age = _zero(pd.DataFrame({c: data[c] for c in age_cols}).fillna(0) / 100.0).to_numpy(dtype="float64")
    h_age = pd.Series((weights_age * weights_age).sum(axis=1), index=data.index)

    type_cols = [
        "RC_M1_SHC_RSD_UE_CLN_RAT",
        "RC_M1_SHC_WP_UE_CLN_RAT",
        "RC_M1_SHC_FLP_UE_CLN_RAT",
    ]
    for col in type_cols:
        if col not in data.columns:
            data[col] = 0.0
    weights_type = _zero(pd.DataFrame({c: data[c] for c in type_cols}).fillna(0) / 100.0).to_numpy(dtype="float64")
    h_type = pd.Series((weights_type * weights_type).sum(axis=1), index=data.index)

    def pos_norm(x: pd.Series) -> pd.Series:
        med = x.rolling(ROLL_WINDOW, MIN_PERIODS).median()
        mad = (x - med).abs().rolling(ROLL_WINDOW, MIN_PERIODS).median()
        z = (x - med) / (1.4826 * mad + EPS)
        return _zero(nz(np.maximum(0.0, z)))

    s_mix = pos_norm(h_age)
    s_type = pos_norm(h_type)

    customer_risk = _zero(0.40 * s_drop + 0.25 * s_loyal + 0.20 * s_acq + 0.10 * s_mix + 0.05 * s_type)
    out = data[["ENCODED_MCT", "TA_YM"]].copy()
    out["Customer_Risk"] = customer_risk
    return out


def compute_market_risk(df: pd.DataFrame, key=("ENCODED_MCT", "TA_YM")) -> pd.DataFrame:
    data = df.sort_values(list(key)).copy()
    h_ind = _zero(nz(to_percent(data.get("M12_SME_RY_ME_MCT_RAT", pd.Series(0.0, index=data.index)))))
    h_biz = _zero(nz(to_percent(data.get("M12_SME_BZN_ME_MCT_RAT", pd.Series(0.0, index=data.index)))))
    s_closure = _zero((h_ind + h_biz) / 2.0)

    u_rev = _zero(nz(1.0 - to_percent(data.get("M1_SME_RY_SAA_RAT", pd.Series(100.0, index=data.index)))))
    u_cnt = _zero(nz(1.0 - to_percent(data.get("M1_SME_RY_CNT_RAT", pd.Series(100.0, index=data.index)))))
    s_underperf = _zero((u_rev + u_cnt) / 2.0)

    age_rank = data["MCT_OPE_MS_CN_RANK"]
    s_age = _zero(nz(4.0 * np.minimum(age_rank, 1.0 - age_rank)))

    market_risk = _zero(0.50 * s_closure + 0.35 * s_underperf + 0.15 * s_age)
    out = data[["ENCODED_MCT", "TA_YM"]].copy()
    out["Market_Risk"] = market_risk
    return out
