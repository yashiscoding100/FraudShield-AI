"""
Production-grade Evaluation Engine for Imbalanced Fraud Detection.
Computes PR-AUC, ROC-AUC, Recall@FPR thresholds, and Cost-Optimal Decision Thresholds.
"""

import os
import json
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

# Optional plotting with fallback
HAVE_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception as e:
    HAVE_MATPLOTLIB = False
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.Evaluate")


def evaluate_test_set(
    risk_engine,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "models/evaluation",
    cost_fn: float = 200.0,  # Average cost of an undetected fraud chargeback
    cost_fp: float = 15.0    # Friction cost of falsely challenging a legitimate user
) -> Dict[str, Any]:
    """
    Evaluates the trained risk engine against unseen test data.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Evaluating model on {len(X_test):,} test transactions ({y_test.sum()} frauds)...")

    # 1. Compute predictions
    preds_df = risk_engine.predict_risk(X_test)
    y_scores = preds_df["calibrated_prob"].values
    risk_scores = preds_df["risk_score"].values

    # 2. Precision-Recall AUC (Primary metric for severe class imbalance)
    pr_auc = float(average_precision_score(y_test, y_scores))
    roc_auc = float(roc_auc_score(y_test, y_scores))

    # 3. Precision-Recall Curve & Optimal F1 Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    
    # Calculate F1 for each threshold (avoiding div by zero)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])

    # 4. Recall at strict False Positive Rate (FPR <= 0.1% = 1 false alarm per 1,000 txns)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
    idx_01_fpr = np.where(fpr <= 0.001)[0]
    recall_at_01_fpr = float(tpr[idx_01_fpr[-1]]) if len(idx_01_fpr) > 0 else 0.0

    # 5. Financial Cost Optimization Curve
    # Find the threshold that minimizes financial loss = (FN * cost_fn) + (FP * cost_fp)
    min_cost = float("inf")
    cost_optimal_threshold = 0.5
    best_cm = None

    for thresh in np.linspace(0.01, 0.99, 100):
        y_pred = (y_scores >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_loss = (fn * cost_fn) + (fp * cost_fp)
        if total_loss < min_cost:
            min_cost = total_loss
            cost_optimal_threshold = float(thresh)
            best_cm = cm

    # Confusion matrix at optimal F1 threshold
    y_pred_opt = (y_scores >= best_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()

    report = classification_report(y_test, y_pred_opt, output_dict=True)

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "best_f1_score": round(best_f1, 4),
        "optimal_f1_threshold": round(best_threshold, 4),
        "precision_at_optimal_threshold": round(best_precision, 4),
        "recall_at_optimal_threshold": round(best_recall, 4),
        "recall_at_0.1pct_fpr": round(recall_at_01_fpr, 4),
        "cost_optimal_threshold": round(cost_optimal_threshold, 4),
        "minimized_financial_cost": round(min_cost, 2),
        "confusion_matrix_optimal": {
            "true_negatives": int(tn_opt),
            "false_positives": int(fp_opt),
            "false_negatives": int(fn_opt),
            "true_positives": int(tp_opt)
        },
        "classification_report": report
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics written to {metrics_path}")

    # Generate & save PR Curve plot if matplotlib is available
    if HAVE_MATPLOTLIB:
        try:
            plt.figure(figsize=(7, 5))
            plt.plot(recalls, precisions, color="#e63946", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
            plt.scatter([best_recall], [best_precision], color="#1d3557", s=80, zorder=5, label=f"Optimal F1 ({best_f1:.3f})")
            plt.title("Precision-Recall Curve (Credit Card Fraud Benchmark)", fontsize=13, fontweight="bold")
            plt.xlabel("Recall (Fraud Detection Rate)", fontsize=11)
            plt.ylabel("Precision (True Positive Accuracy)", fontsize=11)
            plt.legend(loc="lower left")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=200)
            plt.close()

            # Generate & save ROC Curve plot
            plt.figure(figsize=(7, 5))
            plt.plot(fpr, tpr, color="#457b9d", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], color="#a8dadc", linestyle="--")
            plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
            plt.xlabel("False Positive Rate", fontsize=11)
            plt.ylabel("True Positive Rate (Recall)", fontsize=11)
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=200)
            plt.close()
            logger.info(f"Evaluation plots saved to {output_dir}")
        except Exception as e:
            logger.warning(f"Could not render plots: {e}")

    logger.info(f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Optimal F1: {best_f1:.4f} | Recall@0.1% FPR: {recall_at_01_fpr:.4f}")
    return metrics
