"""
Feature Engineering and Preprocessing Pipeline for Credit Card Fraud Detection.
Produces robust, normalized features suitable for both tree ensembles and anomaly detection.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Credit Card Fraud:
    - Log transforms heavily skewed transaction amounts
    - Scales amount with RobustScaler (resistant to extreme outlier spikes)
    - Extracts cyclical hour-of-day features (sine/cosine) from elapsed seconds
    - Generates multi-component anomaly distance metrics from top predictive PCA components
    """

    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.top_pca_cols = ["V4", "V10", "V11", "V12", "V14", "V17"]

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        
        # Fit scaler on log-transformed amounts
        if "Amount" in df.columns:
            log_amount = np.log1p(df[["Amount"]].values)
            self.amount_scaler.fit(log_amount)
            
        transformed_df = self._transform_dataframe(df)
        self.feature_names = list(transformed_df.columns)
        return self

    def _transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        
        # 1. Base PCA features (V1 .. V28)
        pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
        for col in pca_cols:
            out[col] = df[col].astype(np.float32)

        # 2. Amount features
        if "Amount" in df.columns:
            out["Amount_raw"] = df["Amount"].astype(np.float32)
            log_amount = np.log1p(df[["Amount"]].values)
            out["Amount_scaled"] = self.amount_scaler.transform(log_amount).flatten().astype(np.float32)

        # 3. Cyclical Temporal features
        if "Time" in df.columns:
            hour_of_day = (df["Time"] / 3600.0) % 24.0
            out["Hour_of_day"] = hour_of_day.astype(np.float32)
            out["Hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0).astype(np.float32)
            out["Hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0).astype(np.float32)

        # 4. Domain & Interaction Features (Key Fraud Predictors)
        available_top_pca = [col for col in self.top_pca_cols if col in df.columns]
        if available_top_pca:
            # Multi-dimensional deviation: L2 norm of the most sensitive components
            out["PCA_anomaly_magnitude"] = np.sqrt(
                np.sum(df[available_top_pca].values ** 2, axis=1)
            ).astype(np.float32)

        if "V14" in df.columns and "V4" in df.columns:
            out["Interaction_V14_V4"] = (df["V14"] * df["V4"]).astype(np.float32)

        if "V12" in df.columns and "V10" in df.columns:
            out["Interaction_V12_V10"] = (df["V12"] * df["V10"]).astype(np.float32)

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_dataframe(X.copy())
        # Ensure column alignment
        for col in self.feature_names:
            if col not in transformed.columns:
                transformed[col] = 0.0
        return transformed[self.feature_names]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names


def prepare_features_and_target(
    df: pd.DataFrame, 
    pipeline: FraudFeatureEngineer = None, 
    fit: bool = False
) -> Tuple[pd.DataFrame, pd.Series, FraudFeatureEngineer]:
    """
    Extracts features and target, applying the feature engineering pipeline.
    """
    if "Class" in df.columns:
        y = df["Class"].copy()
        X_raw = df.drop(columns=["Class"])
    else:
        y = None
        X_raw = df.copy()

    if pipeline is None:
        pipeline = FraudFeatureEngineer()

    if fit:
        X_proc = pipeline.fit_transform(X_raw)
    else:
        X_proc = pipeline.transform(X_raw)

    return X_proc, y, pipeline


if __name__ == "__main__":
    from data_loader import download_or_load_dataset
    df = download_or_load_dataset()
    X, y, pipe = prepare_features_and_target(df.head(100), fit=True)
    print(f"Features created: {len(pipe.get_feature_names_out())}")
    print(pipe.get_feature_names_out())
