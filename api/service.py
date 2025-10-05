"""Core scoring logic that powers the FastAPI endpoints."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .loader import load_risk_output

ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3

BASE_MARKET = {
    "default": 0.55,
    "치킨": 0.56,
    "카페": 0.54,
    "피자": 0.57,
    "편의점": 0.53,
}


def _market_risk(industry: Optional[str], region: Optional[str], delivery_share: Optional[float]) -> float:
    """Estimate market risk based on industry, region, and delivery preference."""
    base = BASE_MARKET.get(industry, BASE_MARKET["default"])
    if delivery_share is not None:
        base += 0.02 * (delivery_share - 0.5)
    if region and region.endswith("구"):
        base += 0.002
    return float(np.clip(base, 0.45, 0.65))


def _label_alert(p_final: float, green_threshold: float = 0.7, orange_threshold: float = 0.9) -> str:
    if p_final >= orange_threshold:
        return "RED"
    if p_final >= green_threshold:
        return "ORANGE"
    return "GREEN"


def _explain(risk_components: Dict[str, float]) -> list[str]:
    explanations: list[str] = []
    if risk_components.get("Sales_Risk", 0) > 0.05:
        explanations.append("최근 1→3개월 대비 매출 모멘텀 둔화")
    if risk_components.get("Customer_Risk", 0) > 0.05:
        explanations.append("고객 수 감소 신호")
    if risk_components.get("Market_Risk", 0) > 0.55:
        explanations.append("지역/업종 시장 위험도 상회")
    return explanations or ["위험 신호는 크지 않음"]


def compute_rule_risks(
    sales_1m: Optional[float],
    sales_3m_avg: Optional[float],
    cust_1m: Optional[float],
    cust_3m_avg: Optional[float],
    industry_code: Optional[str],
    region_code: Optional[str],
    delivery_share: Optional[float],
) -> Dict[str, float]:
    """Compute rule-based component risks from raw business metrics."""
    sales_risk, cust_risk = 0.0, 0.0
    if sales_1m and sales_3m_avg and sales_3m_avg > 0:
        momentum = (sales_1m - sales_3m_avg) / (sales_3m_avg + 1e-9)
        sales_risk = float(np.clip(-momentum, 0, 1) * 0.1)
    if cust_1m and cust_3m_avg and cust_3m_avg > 0:
        momentum = (cust_1m - cust_3m_avg) / (cust_3m_avg + 1e-9)
        cust_risk = float(np.clip(-momentum, 0, 1) * 0.1)
    market_risk = _market_risk(industry_code, region_code, delivery_share)
    return {
        "Sales_Risk": round(sales_risk, 6),
        "Customer_Risk": round(cust_risk, 6),
        "Market_Risk": round(market_risk, 6),
    }


def blend_final(p_model: Optional[float], risk_components: Dict[str, float]) -> float:
    """Combine rule-based risk and model score into a final probability."""
    risk_score = (
        ALPHA * risk_components["Sales_Risk"]
        + BETA * risk_components["Customer_Risk"]
        + GAMMA * risk_components["Market_Risk"]
    )
    if p_model is None:
        return float(np.clip(risk_score, 0, 1))
    return float(np.clip(0.4 * risk_score + 0.6 * p_model, 0, 1))


def _risk_score(risk_components: Dict[str, float]) -> float:
    return round(
        ALPHA * risk_components["Sales_Risk"]
        + BETA * risk_components["Customer_Risk"]
        + GAMMA * risk_components["Market_Risk"],
        6,
    )


def predict_batch(store_id: Optional[str], target_month: Optional[str]) -> Optional[Dict[str, object]]:
    """Fetch the latest prediction for a store/month combination from disk."""
    df = load_risk_output()
    if df is None:
        return None

    filtered = df.copy()
    if store_id:
        filtered = filtered[filtered["ENCODED_MCT"] == str(store_id)]
    if target_month:
        tm = pd.to_datetime(f"{target_month}-01", errors="coerce")
        filtered = filtered[filtered["TA_YM"] == tm]

    if filtered.empty:
        return None

    row = filtered.sort_values("TA_YM").iloc[-1]
    risk_components = {
        "Sales_Risk": float(row.get("Sales_Risk", 0.0)),
        "Customer_Risk": float(row.get("Customer_Risk", 0.0)),
        "Market_Risk": float(row.get("Market_Risk", 0.0)),
    }
    p_model = float(row.get("p_model", 0.0))
    p_final = float(row.get("p_final", blend_final(p_model, risk_components)))

    return {
        "store_id": store_id,
        "target_month": row["TA_YM"].strftime("%Y-%m"),
        "p_model": p_model,
        "risk_components": risk_components,
        "risk_score": _risk_score(risk_components),
        "p_final": p_final,
        "alert": _label_alert(p_final),
        "explanations": _explain(risk_components),
    }


def quickscore(payload: Dict[str, Optional[object]]) -> Dict[str, object]:
    """Score using rule heuristics only (used as a fallback)."""
    risk_components = compute_rule_risks(
        payload.get("sales_1m"),
        payload.get("sales_3m_avg"),
        payload.get("cust_1m"),
        payload.get("cust_3m_avg"),
        payload.get("industry_code"),
        payload.get("region_code"),
        payload.get("delivery_share"),
    )
    p_final = blend_final(None, risk_components)
    return {
        "store_id": payload.get("store_id"),
        "target_month": payload.get("target_month"),
        "p_model": 0.0,
        "risk_components": risk_components,
        "risk_score": _risk_score(risk_components),
        "p_final": p_final,
        "alert": _label_alert(p_final),
        "explanations": _explain(risk_components),
    }
