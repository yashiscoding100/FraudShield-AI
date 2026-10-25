"""
Model Architecture:
1. Supervised XGBoost Classifier with cost-sensitive class weighting.
2. Isotonic Probability Calibration via Pool-Adjacent-Violators Algorithm (PAVA).
3. Unsupervised Isolation Forest (Liu, Ting, Zhou) + PCA Reconstruction Error for Anomaly Detection.
4. Hybrid Risk Engine producing a unified, tier-based Risk Score (0 - 100).
"""

import os
import joblib
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.Models")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class IsotonicCalibrator:
    """
    Pure NumPy implementation of Isotonic Regression via the 
    Pool Adjacent Violators Algorithm (PAVA). Fits a non-decreasing step function
    to map skewed model probability margins to empirical true fraud rates.
    """

    def __init__(self):
        self.thresholds: np.ndarray = np.array([])
        self.values: np.ndarray = np.array([])

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Sort by raw score
        order = np.argsort(scores)
        x_sorted = scores[order]
        y_sorted = y[order]

        # PAVA block representation: [weight, sum_y, max_x]
        blocks = []
        for xi, yi in zip(x_sorted, y_sorted):
            curr_weight = 1.0
            curr_val = yi
            curr_max_x = xi

            while blocks and (blocks[-1][1] / blocks[-1][0]) >= (curr_val / curr_weight):
                prev_w, prev_y, prev_max_x = blocks.pop()
                curr_weight += prev_w
                curr_val += prev_y

            blocks.append((curr_weight, curr_val, curr_max_x))

        self.thresholds = np.array([b[2] for b in blocks])
        self.values = np.clip(np.array([b[1] / b[0] for b in blocks]), 0.0, 1.0)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if len(self.thresholds) == 0:
            return np.asarray(scores, dtype=np.float64)
        
        scores = np.asarray(scores, dtype=np.float64)
        indices = np.searchsorted(self.thresholds, scores)
        indices = np.clip(indices, 0, len(self.values) - 1)
        return self.values[indices]


class IsolationTreeNode:
    """Node in an Isolation Tree."""
    def __init__(self, left=None, right=None, split_feat=None, split_val=None, size=1):
        self.left = left
        self.right = right
        self.split_feat = split_feat
        self.split_val = split_val
        self.size = size
        self.is_leaf = left is None and right is None


class PureIsolationForest:
    """
    Pure NumPy implementation of Isolation Forest (Liu, Ting, Zhou 2008).
    Isolates anomalies by randomly selecting a feature and a split value.
    Anomalies have shorter average path lengths from the root.
    """

    def __init__(self, n_estimators: int = 100, max_samples: int = 256, max_depth: int = 8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: List[IsolationTreeNode] = []
        self.pca = PCA(n_components=5)
        self.pca_mean_error = 1.0

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 1.0
        if n == 2:
            return 1.0
        euler_gamma = 0.5772156649
        return 2.0 * (np.log(n - 1) + euler_gamma) - (2.0 * (n - 1) / n)

    def _build_tree(self, X: np.ndarray, depth: int, rng: np.random.RandomState) -> IsolationTreeNode:
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples <= 1:
            return IsolationTreeNode(size=n_samples)

        feat_idx = rng.randint(0, n_features)
        feat_min = X[:, feat_idx].min()
        feat_max = X[:, feat_idx].max()

        if feat_min == feat_max:
            return IsolationTreeNode(size=n_samples)

        split_val = rng.uniform(feat_min, feat_max)
        left_mask = X[:, feat_idx] < split_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return IsolationTreeNode(size=n_samples)

        left_child = self._build_tree(X[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], depth + 1, rng)
        return IsolationTreeNode(left=left_child, right=right_child, split_feat=feat_idx, split_val=split_val, size=n_samples)

    def fit(self, X: np.ndarray) -> "PureIsolationForest":
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        sample_size = min(self.max_samples, n_samples)
        
        # 1. Fit Isolation Trees
        self.trees = []
        for _ in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
            tree = self._build_tree(X[sample_idx], depth=0, rng=rng)
            self.trees.append(tree)

        # 2. Fit PCA Reconstruction baseline on normal data
        pca_sample = X[rng.choice(n_samples, size=min(10000, n_samples), replace=False)]
        self.pca.fit(pca_sample)
        reconstructed = self.pca.inverse_transform(self.pca.transform(pca_sample))
        errors = np.sqrt(np.mean((pca_sample - reconstructed) ** 2, axis=1))
        self.pca_mean_error = float(np.mean(errors)) + 1e-6
        return self

    def _path_length(self, x: np.ndarray, node: IsolationTreeNode, current_depth: int) -> float:
        if node.is_leaf:
            return current_depth + self._c(node.size)
        if x[node.split_feat] < node.split_val:
            return self._path_length(x, node.left, current_depth + 1)
        else:
            return self._path_length(x, node.right, current_depth + 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly risk normalized between [0, 1].
        Combines Isolation Tree path compression with PCA reconstruction error.
        """
        X = np.asarray(X, dtype=np.float32)
        c_factor = self._c(self.max_samples)
        
        # 1. Average path length
        path_lengths = np.zeros(len(X))
        for tree in self.trees:
            lengths = np.array([self._path_length(xi, tree, 0) for xi in X])
            path_lengths += lengths
        avg_paths = path_lengths / len(self.trees)
        iso_scores = 2.0 ** (-avg_paths / c_factor)

        # 2. PCA Reconstruction Error
        reconstructed = self.pca.inverse_transform(self.pca.transform(X))
        recon_errors = np.sqrt(np.mean((X - reconstructed) ** 2, axis=1))
        norm_recon = np.clip(recon_errors / (3.0 * self.pca_mean_error), 0.0, 1.0)

        # Blended anomaly index
        composite_anomaly = 0.6 * iso_scores + 0.4 * norm_recon
        return np.clip(composite_anomaly, 0.0, 1.0)


class FraudRiskEngine:
    """
    Ensemble Risk Engine combining Supervised Gradient Boosting with 
    PAVA Probability Calibration and Unsupervised Isolation Forest.
    """

    def __init__(
        self,
        supervised_model: xgb.XGBClassifier = None,
        calibrator: IsotonicCalibrator = None,
        anomaly_model: PureIsolationForest = None,
        supervised_weight: float = 0.75,
        anomaly_weight: float = 0.25,
    ):
        self.supervised_model = supervised_model
        self.calibrator = calibrator
        self.anomaly_model = anomaly_model
        self.supervised_weight = supervised_weight
        self.anomaly_weight = anomaly_weight

    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates calibrated fraud probability, anomaly score, and blended risk score.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # 1. Supervised Calibrated Probability
        if self.supervised_model is not None:
            raw_probs = self.supervised_model.predict_proba(X_arr)[:, 1]
            if self.calibrator is not None:
                calibrated_probs = self.calibrator.predict(raw_probs)
            else:
                calibrated_probs = raw_probs
        else:
            raise ValueError("Supervised model is not trained.")

        # 2. Unsupervised Anomaly Score
        if self.anomaly_model is not None:
            anomaly_risk = self.anomaly_model.score_samples(X_arr)
        else:
            anomaly_risk = np.zeros(len(X_arr))

        # 3. Hybrid Blended Score (0 - 100)
        blended = (
            self.supervised_weight * calibrated_probs +
            self.anomaly_weight * anomaly_risk
        )
        risk_score = np.clip(blended * 100.0, 0.0, 100.0)

        # 4. Categorize Risk Tiers
        risk_tiers = []
        recommendations = []
        for s in risk_score:
            if s >= 80.0:
                risk_tiers.append("CRITICAL")
                recommendations.append("Immediate Account Freeze & Alert")
            elif s >= 55.0:
                risk_tiers.append("HIGH")
                recommendations.append("Route to Human Investigator Queue")
            elif s >= 25.0:
                risk_tiers.append("MEDIUM")
                recommendations.append("Step-up Authentication (3D Secure / OTP)")
            else:
                risk_tiers.append("LOW")
                recommendations.append("Auto-Approve")

        results = pd.DataFrame({
            "risk_score": np.round(risk_score, 2),
            "calibrated_prob": np.round(calibrated_probs, 4),
            "anomaly_risk": np.round(anomaly_risk, 4),
            "risk_tier": risk_tiers,
            "action_recommendation": recommendations
        }, index=X.index if isinstance(X, pd.DataFrame) else None)

        return results

    def save(self, output_dir: str = MODELS_DIR):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.supervised_model, os.path.join(output_dir, "xgboost_supervised.joblib"))
        joblib.dump(self.calibrator, os.path.join(output_dir, "calibrator.joblib"))
        joblib.dump(self.anomaly_model, os.path.join(output_dir, "isolation_forest.joblib"))
        logger.info(f"Models successfully saved to {output_dir}")

    @classmethod
    def load(cls, models_dir: str = MODELS_DIR) -> "FraudRiskEngine":
        sup_path = os.path.join(models_dir, "xgboost_supervised.joblib")
        cal_path = os.path.join(models_dir, "calibrator.joblib")
        ano_path = os.path.join(models_dir, "isolation_forest.joblib")

        supervised_model = joblib.load(sup_path) if os.path.exists(sup_path) else None
        calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
        anomaly_model = joblib.load(ano_path) if os.path.exists(ano_path) else None

        return cls(
            supervised_model=supervised_model,
            calibrator=calibrator,
            anomaly_model=anomaly_model
        )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> FraudRiskEngine:
    """
    Trains the XGBoost model with class imbalance correction,
    calibrates probabilities with PAVA Isotonic Regression on validation data,
    and fits Isolation Forest on normal transactions.
    """
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    logger.info(f"Training XGBoost with scale_pos_weight={scale_pos_weight:.2f} (Neg: {neg_count}, Pos: {pos_count})...")

    # 1. Train Supervised XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    logger.info("XGBoost training complete.")

    # 2. Probability Calibration (PAVA Isotonic Regression on Validation Set)
    logger.info("Fitting PAVA Isotonic Probability Calibrator on validation set...")
    val_raw_probs = xgb_clf.predict_proba(X_val)[:, 1]
    calibrator = IsotonicCalibrator()
    calibrator.fit(val_raw_probs, y_val.values)
    logger.info("Probability calibration complete.")

    # 3. Train Pure Isolation Forest (Unsupervised)
    logger.info("Fitting Pure Isolation Forest on legitimate baseline transactions...")
    normal_train = X_train[y_train == 0].values
    iso_forest = PureIsolationForest(n_estimators=80, max_samples=256, random_state=42)
    iso_forest.fit(normal_train)
    logger.info("Isolation Forest fitting complete.")

    engine = FraudRiskEngine(
        supervised_model=xgb_clf,
        calibrator=calibrator,
        anomaly_model=iso_forest
    )
    return engine
