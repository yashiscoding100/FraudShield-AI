"""
End-to-end Training Pipeline for FraudShield.
Coordinates data loading, feature engineering, model training,
probability calibration, explainability fitting, and artifact serialization.
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import download_or_load_dataset, split_data, PROCESSED_DIR
from features import FraudFeatureEngineer, prepare_features_and_target
from models import train_models, FraudRiskEngine, MODELS_DIR
from explainability import FraudExplainer
from evaluate import evaluate_test_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.TrainPipeline")


def run_training_pipeline() -> dict:
    logger.info("=== STEP 1: Acquiring Benchmark Dataset ===")
    df = download_or_load_dataset()

    logger.info("=== STEP 2: Stratified Train/Val/Test Split ===")
    splits = split_data(df, test_size=0.15, val_size=0.15, random_state=42)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    logger.info("=== STEP 3: Feature Engineering & Preprocessing ===")
    feature_pipeline = FraudFeatureEngineer()
    
    # Fit feature pipeline strictly on training data
    X_train, y_train, _ = prepare_features_and_target(train_df, pipeline=feature_pipeline, fit=True)
    X_val, y_val, _ = prepare_features_and_target(val_df, pipeline=feature_pipeline, fit=False)
    X_test, y_test, _ = prepare_features_and_target(test_df, pipeline=feature_pipeline, fit=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.joblib")
    joblib.dump(feature_pipeline, pipeline_path)
    logger.info(f"Saved feature pipeline ({len(feature_pipeline.feature_names)} features) to {pipeline_path}")

    logger.info("=== STEP 4: Training Models & Probability Calibration ===")
    risk_engine = train_models(X_train, y_train, X_val, y_val)
    risk_engine.save(MODELS_DIR)

    logger.info("=== STEP 5: Fitting TreeSHAP Explainer ===")
    explainer = FraudExplainer(
        model=risk_engine.supervised_model,
        feature_names=feature_pipeline.feature_names
    )
    explainer.save()

    logger.info("=== STEP 6: Rigorous Test Set Evaluation ===")
    metrics = evaluate_test_set(
        risk_engine=risk_engine,
        X_test=X_test,
        y_test=y_test,
        output_dir=os.path.join(MODELS_DIR, "evaluation")
    )

    logger.info("=== Training & Evaluation Pipeline Completed Successfully! ===")
    return metrics


if __name__ == "__main__":
    run_training_pipeline()
