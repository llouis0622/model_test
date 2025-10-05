"""Utility helpers for reading artifacts used by the API service."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1])).resolve()
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RISK_OUTPUT_PATH = BASE_DIR / "risk_output_trained.csv"


def load_risk_output() -> Optional[pd.DataFrame]:
    """Load the precomputed risk output if it exists."""
    if not RISK_OUTPUT_PATH.exists():
        return None
    df = pd.read_csv(RISK_OUTPUT_PATH)
    df["ENCODED_MCT"] = df["ENCODED_MCT"].astype(str)
    df["TA_YM"] = pd.to_datetime(df["TA_YM"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return df
