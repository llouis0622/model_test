"""Combine individual risk components into a single score."""
from __future__ import annotations

import pandas as pd

from config import ALPHA, BETA, GAMMA
from risk_components import compute_customer_risk, compute_market_risk, compute_sales_risk


def compute_all_risks(df: pd.DataFrame) -> pd.DataFrame:
    sales = compute_sales_risk(df)
    customer = compute_customer_risk(df)
    market = compute_market_risk(df)

    out = sales.merge(customer, on=["ENCODED_MCT", "TA_YM"], how="left")
    out = out.merge(market, on=["ENCODED_MCT", "TA_YM"], how="left")

    out["RiskScore"] = (
        ALPHA * out["Sales_Risk"].fillna(0)
        + BETA * out["Customer_Risk"].fillna(0)
        + GAMMA * out["Market_Risk"].fillna(0)
    )
    return out
