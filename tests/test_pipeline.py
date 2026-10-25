"""
Unit and integration tests for FraudShield feature pipeline and models.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from features import FraudFeatureEngineer, prepare_features_and_target
from models import FraudRiskEngine, train_models


@pytest.fixture
def sample_transaction_df():
    """Generates synthetic transactions matching the benchmark schema."""
    np.random.seed(42)
    n = 200
    data = {
        "Time": np.random.uniform(0, 172800, n),
        "Amount": np.random.exponential(scale=88.0, size=n) + 1.0,
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1.5, n)
    
    # Injected 5 frauds
    classes = np.zeros(n, dtype=int)
    classes[:5] = 1
    data["Class"] = classes
    
    return pd.DataFrame(data)


def test_feature_pipeline_fit_transform(sample_transaction_df):
    pipeline = FraudFeatureEngineer()
    X, y, pipe = prepare_features_and_target(sample_transaction_df, pipeline=pipeline, fit=True)
    
    # Assertions
    assert len(X) == len(sample_transaction_df)
    assert not X.isnull().values.any(), "Transformed features must not contain NaN values"
    assert "Amount_scaled" in X.columns
    assert "Hour_sin" in X.columns
    assert "PCA_anomaly_magnitude" in X.columns
    assert "Interaction_V14_V4" in X.columns


def test_feature_pipeline_transform_consistency(sample_transaction_df):
    pipeline = FraudFeatureEngineer()
    pipeline.fit(sample_transaction_df.drop(columns=["Class"]))
    
    # Test single row inference
    single_row = sample_transaction_df.iloc[[0]].drop(columns=["Class"])
    trans_single = pipeline.transform(single_row)
    
    assert len(trans_single) == 1
    assert list(trans_single.columns) == pipeline.feature_names


def test_model_training_and_bounds(sample_transaction_df):
    pipeline = FraudFeatureEngineer()
    X, y, _ = prepare_features_and_target(sample_transaction_df, pipeline=pipeline, fit=True)
    
    # Mock split
    train_idx = list(range(0, 150))
    val_idx = list(range(150, 200))
    
    engine = train_models(
        X_train=X.iloc[train_idx],
        y_train=y.iloc[train_idx],
        X_val=X.iloc[val_idx],
        y_val=y.iloc[val_idx]
    )
    
    preds = engine.predict_risk(X.iloc[val_idx])
    
    assert len(preds) == len(val_idx)
    assert preds["risk_score"].between(0.0, 100.0).all()
    assert preds["calibrated_prob"].between(0.0, 1.0).all()
    assert preds["anomaly_risk"].between(0.0, 1.0).all()
    assert set(preds["risk_tier"].unique()).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
