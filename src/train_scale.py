"""
Large-Scale Training & High-Throughput Benchmark on 1,000,000+ Transactions.
Demonstrates memory downcasting, feature engineering, cost-sensitive gradient boosting,
and batch inference throughput metrics at enterprise scale.
"""

import os
import sys
import time
import json
import logging
from typing import Tuple, List, Dict, Any
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

from models import IsotonicCalibrator, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.ScaleBenchmark")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "million_stream.pkl")


def engineer_scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    High-speed vectorized feature engineering for 1M financial records.
    Engineers financial balance errors, velocity interactions, and categorical dummies.
    """
    logger.info(f"Engineering features for {len(df):,} transactions...")
    t0 = time.time()

    y = df["isFraud"].astype(np.int8)
    
    # 1. Categorical Encodings (One-Hot)
    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=np.float32)
    device_dummies = pd.get_dummies(df["device_type"], prefix="device", dtype=np.float32)

    # 2. Financial Balance Accounting Invariants
    # In legitimate transactions: oldbalanceOrg - amount == newbalanceOrig
    balance_error_orig = ((df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]).astype(np.float32)
    balance_drain_ratio = ((df["oldbalanceOrg"] - df["newbalanceOrig"]) / (df["oldbalanceOrg"] + 1.0)).astype(np.float32)
    amount_to_balance_ratio = (df["amount"] / (df["oldbalanceOrg"] + 1.0)).astype(np.float32)

    # 3. Cyclical Temporal Features
    hour_sin = np.sin(2 * np.pi * df["hour_of_day"] / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * df["hour_of_day"] / 24.0).astype(np.float32)

    # 4. Behavioral Risk Interactions
    velocity_pin_risk = (df["velocity_1h"] * (df["failed_pin_attempts"] + 1)).astype(np.float32)
    geo_amount_velocity = (df["dist_from_home_km"] * np.log1p(df["amount"])).astype(np.float32)

    # Combine into memory-optimized DataFrame
    X = pd.concat([
        pd.DataFrame({
            "amount": np.log1p(df["amount"]).astype(np.float32),
            "oldbalanceOrg": np.log1p(df["oldbalanceOrg"]).astype(np.float32),
            "newbalanceOrig": np.log1p(df["newbalanceOrig"]).astype(np.float32),
            "dist_from_home_km": df["dist_from_home_km"],
            "velocity_1h": df["velocity_1h"].astype(np.float32),
            "failed_pin_attempts": df["failed_pin_attempts"].astype(np.float32),
            "balance_error_orig": balance_error_orig,
            "balance_drain_ratio": balance_drain_ratio,
            "amount_to_balance_ratio": amount_to_balance_ratio,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "velocity_pin_risk": velocity_pin_risk,
            "geo_amount_velocity": geo_amount_velocity,
        }, index=df.index),
        type_dummies,
        device_dummies
    ], axis=1)

    elapsed = time.time() - t0
    logger.info(f"Engineered {X.shape[1]} features across {len(X):,} rows in {elapsed:.2f}s.")
    return X, y


def run_million_scale_training():
    if not os.path.exists(DATA_PATH):
        from generator import generate_million_stream
        df = generate_million_stream(1_000_000)
        df.to_pickle(DATA_PATH)
    else:
        logger.info(f"Loading 1M stream from {DATA_PATH}...")
        df = pd.read_pickle(DATA_PATH)

    # 1. Feature Engineering
    X, y = engineer_scale_features(df)

    # 2. Stratified 70/30 Train/Test Split (700,000 train / 300,000 test)
    logger.info("Performing 70/30 stratified train/test split on 1,000,000 rows...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    logger.info(f"Train set: {len(X_train):,} ({y_train.sum()} frauds) | Test set: {len(X_test):,} ({y_test.sum()} frauds)")

    # 3. Train XGBoost on 700,000 rows
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    logger.info(f"=== TRAINING XGBOOST AT ENTERPRISE SCALE (700,000 ROWS) ===")
    logger.info(f"Configuring tree_method='hist', scale_pos_weight={scale_pos_weight:.2f}...")
    
    t_train_start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    t_train = time.time() - t_train_start
    logger.info(f"XGBoost training on 700,000 rows COMPLETED in {t_train:.2f} seconds!")

    # 4. High-Throughput Batch Inference Benchmark on 300,000 Test Records
    logger.info("=== EVALUATING ON 300,000 UNSEEN TEST TRANSACTIONS ===")
    t_infer_start = time.time()
    y_probs = model.predict_proba(X_test)[:, 1]
    t_infer = time.time() - t_infer_start
    throughput = len(X_test) / max(t_infer, 1e-4)

    logger.info(f"Scored {len(X_test):,} transactions in {t_infer:.2f}s ({throughput:,.0f} transactions/second)!")

    # 5. Compute Benchmark Metrics
    pr_auc = float(average_precision_score(y_test, y_probs))
    roc_auc = float(roc_auc_score(y_test, y_probs))

    y_pred = (y_probs >= 0.50).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"=== SCALE BENCHMARK RESULTS (1,000,000 ROWS) ===")
    logger.info(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")
    logger.info(f"Inference Throughput: {throughput:,.0f} txns/sec")

    # 6. Save Model and Metrics
    scale_eval_dir = os.path.join(MODELS_DIR, "evaluation")
    os.makedirs(scale_eval_dir, exist_ok=True)
    
    scale_metrics = {
        "dataset_name": "PaySim / Sparkov 1,000,000 Transaction Scale Benchmark",
        "total_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "training_time_seconds": round(t_train, 2),
        "batch_inference_time_seconds": round(t_infer, 2),
        "inference_throughput_txns_per_sec": round(throughput, 1),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }

    metrics_path = os.path.join(scale_eval_dir, "metrics_1M.json")
    with open(metrics_path, "w") as f:
        json.dump(scale_metrics, f, indent=2)
    logger.info(f"Scale metrics saved to {metrics_path}")

    # Save model
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_scale_1M.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "scale_features.joblib"))
    logger.info("Scale model artifacts saved successfully.")

    return scale_metrics


if __name__ == "__main__":
    run_million_scale_training()
