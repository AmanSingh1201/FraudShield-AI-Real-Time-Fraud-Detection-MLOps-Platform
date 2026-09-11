"""Phase 9: FastAPI real-time inference service."""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.fraudshield.explainability.shap_explainer import FraudExplainer
from src.fraudshield.models.inference_pipeline import FraudInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraudshield")

app = FastAPI(title="FraudShield AI", version="0.1.0",
              description="Real-time fraud detection inference service")

_pipeline: Optional[FraudInferencePipeline] = None
_explainer: Optional[FraudExplainer] = None
_metrics = {"n_requests": 0, "n_errors": 0, "n_block": 0, "n_review": 0, "n_approve": 0}


@app.on_event("startup")
def load_model():
    global _pipeline, _explainer
    _pipeline = FraudInferencePipeline()
    _explainer = FraudExplainer()
    logger.info(f"Model loaded: version={_pipeline.model_version} threshold={_pipeline.threshold}")


class Transaction(BaseModel):
    transaction_id: str = Field(..., description="Client-supplied transaction identifier")
    Time: float
    Amount: float = Field(..., ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

    @field_validator("Amount")
    @classmethod
    def amount_sane(cls, v):
        if v > 1_000_000:
            raise ValueError("Amount exceeds sane upper bound")
        return v


class BatchRequest(BaseModel):
    transactions: list[Transaction]


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _metrics["n_errors"] += 1
        logger.exception(f"request_id={request_id} unhandled_error")
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": request_id})
    response.headers["X-Request-ID"] = request_id
    logger.info(f"request_id={request_id} path={request.url.path} "
                f"status={response.status_code} latency_ms={(time.perf_counter()-t0)*1000:.2f}")
    return response


@app.get("/health")
def health():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": _pipeline.model_version}


@app.get("/model/info")
def model_info():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "model_version": _pipeline.model_version,
        "decision_threshold": _pipeline.threshold,
        "loaded_at": _pipeline.loaded_at,
    }


@app.get("/metrics")
def metrics():
    return _metrics


@app.post("/predict")
def predict(tx: Transaction):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        result = _pipeline.predict_one(tx.model_dump(exclude={"transaction_id"}))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        _metrics["n_errors"] += 1
        logger.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="prediction failed")

    _metrics["n_requests"] += 1
    _metrics[f"n_{result['decision'].lower()}"] = _metrics.get(f"n_{result['decision'].lower()}", 0) + 1
    return {"transaction_id": tx.transaction_id, **result}


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    results = []
    for tx in batch.transactions:
        try:
            r = _pipeline.predict_one(tx.model_dump(exclude={"transaction_id"}))
            _metrics["n_requests"] += 1
            results.append({"transaction_id": tx.transaction_id, **r})
        except Exception:
            _metrics["n_errors"] += 1
            results.append({"transaction_id": tx.transaction_id, "error": "prediction_failed"})
    return {"results": results}


@app.post("/explain")
def explain(tx: Transaction):
    if _explainer is None:
        raise HTTPException(status_code=503, detail="explainer not loaded")
    import pandas as pd
    row = pd.DataFrame([tx.model_dump(exclude={"transaction_id"})])
    try:
        explanation = _explainer.explain_row(row)
    except Exception:
        logger.exception("explain_failed")
        raise HTTPException(status_code=500, detail="explanation failed")
    pred = _pipeline.predict_one(tx.model_dump(exclude={"transaction_id"}))
    return {"transaction_id": tx.transaction_id, "fraud_probability": pred["fraud_probability"],
            "decision": pred["decision"], **explanation}
