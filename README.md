# 🛡️ FraudShield: AI-Powered Fraud Detection Engine

**FraudShield** is an end-to-end, production-grade fraud detection platform built for FinTech applications. It addresses the fundamental challenges of transaction fraud: **extreme class imbalance (0.172%)**, **adversarial distribution shifts**, **strict latency constraints (<50ms)**, and **explainability for compliance and fraud operations teams**.

---

## 🏗️ System Architecture

```
                                  ┌────────────────────────────────────────┐
                                  │   Incoming Transaction Stream          │
                                  └──────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                  ┌────────────────────────────────────────┐
                                  │   Feature Engineering & Scaling        │
                                  │   - Log RobustScaler                   │
                                  │   - Cyclical Hour Embeddings           │
                                  │   - Multi-component L2 Anomaly Norm   │
                                  └──────────┬───────────────────┬─────────┘
                                             │                   │
                        ┌────────────────────┘                   └────────────────────┐
                        ▼                                                             ▼
         ┌──────────────────────────────┐                              ┌──────────────────────────────┐
         │  Supervised XGBoost Engine   │                              │ Unsupervised Isolation Forest│
         │  - scale_pos_weight tuning   │                              │ - Fit on legitimate baseline │
         │  - Isotonic Prob Calibration │                              │ - Out-of-distribution flags  │
         └──────────────┬───────────────┘                              └──────────────┬───────────────┘
                        │                                                             │
                        │ P(Fraud)                                      Anomaly Score │
                        └────────────────────┬───────────────────┬────────────────────┘
                                             ▼                   ▼
                                  ┌────────────────────────────────────────┐
                                  │         Hybrid Risk Engine             │
                                  │  Risk = α · P(Fraud) + β · Anomaly     │
                                  │  Output: Score (0-100) & Action Tier   │
                                  └──────────────────┬─────────────────────┘
                                                     │
                        ┌────────────────────────────┴─────────────────────────────┐
                        ▼                                                          ▼
         ┌──────────────────────────────┐                           ┌──────────────────────────────┐
         │     TreeSHAP Attribution     │                           │   Investigator AI Dossier    │
         │   Extracts local feature     │                           │  Synthesizes natural-language│
         │   attributions per txn       │                           │  briefings & action checklist│
         └──────────────┬───────────────┘                           └──────────────┬───────────────┘
                        │                                                          │
                        └────────────────────────────┬─────────────────────────────┘
                                                     ▼
                                  ┌────────────────────────────────────────┐
                                  │   FastAPI Microservice & Dashboard     │
                                  │   - Sub-50ms REST Endpoint             │
                                  │   - Streamlit Operations Center        │
                                  └────────────────────────────────────────┘
```

---

## 📊 Benchmark Evaluation Results (Unseen Test Set: 42,722 Transactions)

| Metric | FraudShield Performance | Industry / Interview Context |
| :--- | :--- | :--- |
| **PR-AUC (Precision-Recall AUC)** | **0.8081** | Primary metric for 0.172% extreme class imbalance |
| **ROC-AUC** | **0.9749** | Overall ranking capability across all thresholds |
| **Optimal F1-Score** | **0.8194** | Harmonic mean of Precision (84.3%) and Recall (79.7%) |
| **Recall @ 0.1% FPR** | **85.14%** | Catches 85.1% of all fraud attacks with $\le 1$ false alarm per 1,000 txns |
| **False Positive Rate (FPR)** | **0.025%** | Only 11 false alarms across 42,648 legitimate test transactions |

### Test Set Confusion Matrix (at optimal $F_1$ threshold = 0.40)
* **True Negatives (Legitimate Approved):** 42,637
* **True Positives (Fraud Blocked):** 59 (out of 74 total test frauds)
* **False Positives (Legitimate Challenged):** 11
* **False Negatives (Missed Fraud):** 15

---

## ⚡ Multi-Million Scale Benchmark (1,000,000 Transactions)

In addition to the natural European benchmark, FraudShield includes an enterprise-scale synthetic stream generator implementing the **PaySim / Sparkov multi-agent architecture** (modeling account takeover, card testing, and balance accounting invariants).

| Benchmark Parameter | Result | Engineering Impact |
| :--- | :--- | :--- |
| **Total Stream Size** | **1,000,000 records** (1,500 frauds, 0.15% imbalance) | 156 MB in memory |
| **Train / Test Split** | **700,000 train** / **300,000 test** | Stratified split |
| **Feature Engineering Latency** | **0.33 seconds** | Vectorized NumPy balance invariants & encodings |
| **XGBoost Training Time (700k rows)** | **6.59 seconds** | High-performance `tree_method="hist"` with parallel CPU workers |
| **Test Batch Inference Throughput** | **2,366,392 txns / second** | Scored 300,000 test transactions in **0.13 seconds** |
| **Test Recall @ 0.50 Threshold** | **100.0%** (450 / 450 frauds caught) | Only 1 false positive across 299,550 legitimate transactions |

---

## 🎯 Key Engineering Innovations

### 1. Handling Extreme Class Imbalance (0.172%)
- In the European Credit Card benchmark (284,807 txns, 492 frauds), a naive classifier predicting non-fraud achieves **99.828% accuracy while missing every single fraudster**.
- **FraudShield** rejects accuracy entirely in favor of **PR-AUC (Precision-Recall AUC)** and **Recall @ 0.1% False Positive Rate (FPR)**.
- Models are trained using cost-sensitive learning (`scale_pos_weight = count(neg) / count(pos)`), forcing the gradient boosting trees to prioritize minority positive classes during split finding.

### 2. Isotonic Probability Calibration
- Models trained with extreme class weighting produce distorted predicted probabilities (skewed heavily towards 1.0).
- FraudShield applies post-hoc **Isotonic Regression** on a held-out validation fold. A calibrated output of `0.75` guarantees that historically, 75% of such transactions were fraudulent.

### 3. Hybrid Supervised + Anomaly Detection
- Supervised models catch **known historical attack patterns**.
- Isolation Forests identify **novel, zero-day attacks** (e.g. abnormal transaction vectors that do not match past training examples).

### 4. TreeSHAP Local Explainability
- For every flagged transaction, TreeSHAP computes the exact Shapley contribution of every feature in $O(TLD^2)$ time.
- Operations teams see the top positive drivers pushing a transaction into review rather than a black-box decision.

### 5. Automated Investigator Dossiers & Compliance RAG
- **RAG Architecture:** Embeds internal FinTech compliance policies (Bank Secrecy Act / AML, Regulation E 12 CFR Part 1005, FinCEN SAR mandatory reporting rules under 31 CFR Section 1020.320, and Account Takeover incident playbooks).
- **Automated Filing Synthesis:** When high risk is detected, the semantic vector retriever pulls the exact legal statutes, checks SAR thresholds, and synthesizes an audit-ready compliance report with explicit filing deadlines (e.g. 30 calendar days for FinCEN Form 111).

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and navigate
cd FraudShield

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run the End-to-End Training Pipeline
Downloads the OpenML benchmark, splits stratified folds, trains models, calibrates probabilities, fits TreeSHAP, and computes test metrics:
```bash
python src/train.py
```

### 3. Start the FastAPI Microservice
```bash
python -m uvicorn src.api:app --reload --port 8000
```
- Interactive Swagger API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Model metrics: `http://localhost:8000/metrics`

### 4. Launch the Operations Console (Streamlit)
```bash
streamlit run dashboard/app.py
```
Test real benchmark frauds against normal transactions, inspect SHAP waterfall plots, and view live investigator dossiers.

---

## 🧪 Testing Suite
```bash
pytest tests/ -v
```
Verifies feature pipeline invariants, null safety, probability bounds, and API schema compliance.
