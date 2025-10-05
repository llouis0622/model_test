"""Very lightweight heuristics for parsing Korean utterances."""
from __future__ import annotations

import re
from typing import Dict, Optional

INDUSTRY_MAP = {
    "치킨": ["치킨", "치킨집", "후라이드", "양념치킨"],
    "카페": ["카페", "커피", "디저트", "베이커리"],
    "피자": ["피자"],
    "편의점": ["편의점", "CVS"],
}


def _normalize_currency(text: str) -> Optional[float]:
    cleaned = text.replace(",", "").strip()
    match = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*(원|만원|천만원|억원)?", cleaned)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2) or "원"
    multiplier = {"원": 1, "만원": 1e4, "천만원": 1e7, "억원": 1e8}.get(unit, 1)
    return value * multiplier


def _extract_money(sentence: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    one_month = re.search(
        r"(한\s*달|지난달|최근\s*1개월)[^\d]*(\d[\d,\.]*\s*(원|만원|천만원|억원))",
        sentence,
    )
    if one_month:
        results["sales_1m"] = _normalize_currency(one_month.group(2))

    three_month_avg = re.search(
        r"(3개월|최근\s*세\s*달|최근\s*3개월)[^\d]*(\d[\d,\.]*\s*(원|만원|천만원|억원))",
        sentence,
    )
    if three_month_avg:
        results["sales_3m_avg"] = _normalize_currency(three_month_avg.group(2))

    cust_one_month = re.search(r"(지난달|최근\s*1개월)[^\d]*(\d{1,6})\s*명", sentence)
    if cust_one_month:
        results["cust_1m"] = float(cust_one_month.group(2))

    cust_three_month = re.search(r"(3개월|최근\s*세\s*달|최근\s*3개월)[^\d]*(\d{1,6})\s*명", sentence)
    if cust_three_month:
        results["cust_3m_avg"] = float(cust_three_month.group(2))

    if "sales_1m" not in results:
        fallback = re.search(r"매출[^\d]*(\d[\d,\.]*\s*(원|만원|천만원|억원))", sentence)
        if fallback:
            results["sales_1m"] = _normalize_currency(fallback.group(1))

    return {key: value for key, value in results.items() if value is not None}


def _extract_industry(sentence: str) -> Optional[str]:
    for label, keywords in INDUSTRY_MAP.items():
        if any(keyword in sentence for keyword in keywords):
            return label
    return None


def _extract_region(sentence: str) -> Optional[str]:
    match = re.search(r"([가-힣A-Za-z0-9]+(구|동|시|군|읍|면))", sentence)
    return match.group(1) if match else None


def _extract_delivery_share(sentence: str) -> Optional[float]:
    if "배달 위주" in sentence or "배달 중심" in sentence:
        return 0.8
    if "포장 위주" in sentence or "테이크아웃 위주" in sentence:
        return 0.6
    if "홀 위주" in sentence or "내점 위주" in sentence:
        return 0.3

    ratio = re.search(r"(배달|딜리버리)[^\d]*(\d{1,3})\s*%", sentence)
    if ratio:
        value = int(ratio.group(2))
        return max(0.0, min(1.0, value / 100.0))
    return None


def parse_utterance(utterance: str) -> Dict[str, float | str]:
    """Parse raw text into a structured dictionary expected by the API."""
    sentence = utterance.strip()
    parsed: Dict[str, float | str] = {}
    parsed.update(_extract_money(sentence))

    industry = _extract_industry(sentence)
    if industry:
        parsed["industry_code"] = industry

    region = _extract_region(sentence)
    if region:
        parsed["region_code"] = region

    delivery_share = _extract_delivery_share(sentence)
    if delivery_share is not None:
        parsed["delivery_share"] = delivery_share

    return parsed
