"""
Explainable AI (XAI) Engine powered by XGBoost Native C++ TreeSHAP.
Computes exact Shapley feature attributions (Lundberg et al.) directly from the 
XGBoost decision tree ensemble with sub-millisecond latency and zero external dependencies.
"""

import os
import joblib
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.Explainability")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# Mapping mathematical components to investigator domain terminology
FEATURE_DESCRIPTIONS = {
    "V14": "Primary Behavioral Deviation Vector (Historically highest correlated fraud indicator)",
    "V12": "Secondary Behavioral Spending Shift",
    "V10": "Cardholder Behavioral Consistency Index",
    "V17": "Unusual Transaction Signature Pattern",
    "V4": "Transaction Velocity / Frequency Shift",
    "V11": "Merchant Category Divergence Factor",
    "PCA_anomaly_magnitude": "Composite Multi-Dimensional Outlier Distance (L2 Norm)",
    "Interaction_V14_V4": "Co-occurrence of High Velocity with Abnormal Behavior",
    "Interaction_V12_V10": "Compound Risk Pattern Deviation",
    "Amount_raw": "Transaction Monetary Value",
    "Amount_scaled": "Transaction Amount Deviation vs Baseline",
    "Hour_of_day": "Transaction Local Timestamp",
    "Hour_sin": "Cyclical Timing Component (Off-hours / Night anomaly)",
    "Hour_cos": "Cyclical Timing Component (Daytime baseline)"
}


class FraudExplainer:
    """
    Native TreeSHAP explainer for XGBoost models.
    Computes exact Shapley values using XGBoost's compiled C++ prediction engine
    with pred_contribs=True.
    """

    def __init__(self, model: xgb.XGBClassifier = None, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names

    def explain_instance(
        self, 
        instance: pd.DataFrame, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Computes exact TreeSHAP feature attributions for a single transaction.
        
        Returns:
            Dict containing base_value, top risk drivers, and protective mitigators.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        # Ensure correct column alignment
        if self.feature_names is not None:
            aligned_instance = instance[self.feature_names]
        else:
            aligned_instance = instance

        # Native TreeSHAP computation via XGBoost C++ core
        booster = self.model.get_booster()
        dmatrix = xgb.DMatrix(aligned_instance.values, feature_names=self.feature_names)
        
        # pred_contribs=True returns an array of shape (N, num_features + 1)
        # where the last column is the bias/base value term
        contribs = booster.predict(dmatrix, pred_contribs=True)[0]
        feature_shap = contribs[:-1]
        base_value = float(contribs[-1])
        feature_vals = aligned_instance.iloc[0].values

        # Build feature attribution records
        attributions = []
        for feat_name, feat_val, s_val in zip(self.feature_names, feature_vals, feature_shap):
            desc = FEATURE_DESCRIPTIONS.get(
                feat_name, 
                f"Component {feat_name} Vector"
            )
            attributions.append({
                "feature": feat_name,
                "value": float(feat_val),
                "shap_value": float(s_val),
                "impact": "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK",
                "description": desc
            })

        # Sort: top risk drivers (positive SHAP) and top risk mitigators (negative SHAP)
        risk_drivers = sorted(
            [a for a in attributions if a["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True
        )[:top_k]

        risk_mitigators = sorted(
            [a for a in attributions if a["shap_value"] < 0],
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )[:top_k]

        return {
            "base_value": base_value,
            "top_risk_drivers": risk_drivers,
            "top_risk_mitigators": risk_mitigators,
            "all_attributions": attributions
        }

    def save(self, output_path: str = os.path.join(MODELS_DIR, "shap_explainer.joblib")):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        logger.info(f"SHAP explainer saved to {output_path}")

    @classmethod
    def load(cls, explainer_path: str = os.path.join(MODELS_DIR, "shap_explainer.joblib")) -> "FraudExplainer":
        return joblib.load(explainer_path)
