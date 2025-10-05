"""FastAPI application exposing SME risk scoring endpoints."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import schemas
from .nlp import parse_utterance
from .service import predict_batch, quickscore

app = FastAPI(title="SME Early Warning API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/nlp/parse", response_model=schemas.ParseResponse)
def nlp_parse(req: schemas.ParseRequest) -> schemas.ParseResponse:
    """Extract structured features from a natural language utterance."""
    return schemas.ParseResponse(**parse_utterance(req.utterance))


@app.post("/quickscore", response_model=schemas.PredictResponse)
def quick_score(req: schemas.PredictRequest) -> schemas.PredictResponse:
    """Return a rule-based score when full model predictions are unavailable."""
    return schemas.PredictResponse(**quickscore(req.dict()))


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest) -> schemas.PredictResponse:
    """Return model predictions if present, otherwise fall back to quickscore."""
    response = predict_batch(req.store_id, req.target_month)
    if response is None:
        if any(
            [
                req.sales_1m,
                req.sales_3m_avg,
                req.cust_1m,
                req.cust_3m_avg,
            ]
        ):
            return schemas.PredictResponse(**quickscore(req.dict()))
        raise HTTPException(status_code=404, detail="No prediction found.")
    return schemas.PredictResponse(**response)


@app.get("/healthz")
def health() -> dict[str, str | None]:
    """Simple liveness probe."""
    return {"status": "ok"}
