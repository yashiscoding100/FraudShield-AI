"""
Production FastAPI Microservice for Real-Time Fraud Detection & SHAP Explanations.
Provides sub-50ms inference, calibrated probabilities, anomaly flags, and investigator briefs.
"""

import os
import sys
import uuid
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from models import FraudRiskEngine, MODELS_DIR
from features import FraudFeatureEngineer
from explainability import FraudExplainer
from investigator_llm import generate_llm_dossier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.API")

# Global state for loaded models
models_cache: Dict[str, Any] = {}


def ensure_models_loaded():
    """Ensures model artifacts are loaded in memory."""
    if "pipeline" not in models_cache or "engine" not in models_cache:
        try:
            pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.joblib")
            if os.path.exists(pipeline_path):
                models_cache["pipeline"] = joblib.load(pipeline_path)

            models_cache["engine"] = FraudRiskEngine.load(MODELS_DIR)

            explainer_path = os.path.join(MODELS_DIR, "shap_explainer.joblib")
            if os.path.exists(explainer_path):
                models_cache["explainer"] = FraudExplainer.load(explainer_path)
            logger.info("Models loaded via ensure_models_loaded().")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads machine learning artifacts into memory on server startup.
    """
    logger.info("Starting FraudShield API server and loading models...")
    ensure_models_loaded()
    yield
    logger.info("Shutting down FraudShield API.")


app = FastAPI(
    title="FraudShield AI Engine",
    description="Real-time hybrid fraud detection with TreeSHAP explainability and automated investigator dossiers.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class TransactionPayload(BaseModel):
    transaction_id: Optional[str] = Field(default_factory=lambda: f"TXN-{uuid.uuid4().hex[:8].upper()}")
    Time: float = Field(default=0.0, description="Seconds elapsed since reference start")
    Amount: float = Field(..., gt=0, description="Transaction amount in USD")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class BatchPayload(BaseModel):
    transactions: List[TransactionPayload]


class RiskDriver(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str
    description: str


class PredictionResult(BaseModel):
    transaction_id: str
    risk_score: float
    risk_tier: str
    calibrated_fraud_probability: float
    anomaly_score: float
    action_recommendation: str
    top_risk_drivers: List[RiskDriver]
    investigator_dossier: Dict[str, Any]


@app.get("/health")
def health_check():
    loaded = "engine" in models_cache and models_cache["engine"].supervised_model is not None
    return {
        "status": "healthy" if loaded else "degraded",
        "models_loaded": loaded,
        "feature_pipeline": "pipeline" in models_cache,
        "explainer_ready": "explainer" in models_cache
    }


@app.get("/metrics")
def get_metrics():
    metrics_path = os.path.join(MODELS_DIR, "evaluation", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Evaluation metrics not found. Please run training first.")
    with open(metrics_path, "r") as f:
        import json
        return json.load(f)


@app.post("/predict", response_model=PredictionResult)
def predict_transaction(tx: TransactionPayload, include_llm_summary: bool = Query(default=True)):
    ensure_models_loaded()
    if "pipeline" not in models_cache or "engine" not in models_cache:
        raise HTTPException(status_code=503, detail="Models are not initialized. Run training first.")

    # 1. Convert to DataFrame
    input_data = tx.model_dump()
    txn_id = input_data.pop("transaction_id")
    raw_df = pd.DataFrame([input_data])

    # 2. Transform features
    pipe = models_cache["pipeline"]
    proc_df = pipe.transform(raw_df)

    # 3. Model scoring
    engine: FraudRiskEngine = models_cache["engine"]
    res_df = engine.predict_risk(proc_df)
    row = res_df.iloc[0]

    # 4. SHAP explainability
    top_drivers = []
    if "explainer" in models_cache:
        explainer: FraudExplainer = models_cache["explainer"]
        explanation = explainer.explain_instance(proc_df, top_k=4)
        top_drivers = [RiskDriver(**d) for d in explanation["top_risk_drivers"]]

    # 5. Investigator Dossier
    time_hour = (tx.Time / 3600.0) % 24.0
    dossier = generate_llm_dossier(
        transaction_id=txn_id,
        amount=tx.Amount,
        time_hour=time_hour,
        risk_score=float(row["risk_score"]),
        calibrated_prob=float(row["calibrated_prob"]),
        anomaly_risk=float(row["anomaly_risk"]),
        risk_tier=str(row["risk_tier"]),
        recommendation=str(row["action_recommendation"]),
        top_risk_drivers=[d.model_dump() for d in top_drivers]
    )

    return PredictionResult(
        transaction_id=txn_id,
        risk_score=float(row["risk_score"]),
        risk_tier=str(row["risk_tier"]),
        calibrated_fraud_probability=float(row["calibrated_prob"]),
        anomaly_score=float(row["anomaly_risk"]),
        action_recommendation=str(row["action_recommendation"]),
        top_risk_drivers=top_drivers,
        investigator_dossier=dossier
    )


@app.post("/predict/batch")
def predict_batch(payload: BatchPayload):
    if "pipeline" not in models_cache or "engine" not in models_cache:
        raise HTTPException(status_code=503, detail="Models are not initialized.")

    records = [t.model_dump() for t in payload.transactions]
    txn_ids = [r.pop("transaction_id") for r in records]
    raw_df = pd.DataFrame(records)

    pipe = models_cache["pipeline"]
    proc_df = pipe.transform(raw_df)

    engine: FraudRiskEngine = models_cache["engine"]
    res_df = engine.predict_risk(proc_df)
    res_df["transaction_id"] = txn_ids

    return res_df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
