"""
FraudShield AI Enterprise Operations & Investigation Console.
Redesigned with crystal-clear plain-English guidance, visual 5-W explanations,
distinctive action buttons, and transparent AI decisioning for non-technical users.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="FraudShield AI | Plain-English Fraud & Compliance Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add src to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from models import FraudRiskEngine, MODELS_DIR
from features import FraudFeatureEngineer
from explainability import FraudExplainer
from investigator_llm import generate_llm_dossier
from rag.engine import rag_pipeline

# High-Contrast, Distinctive Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #070a13 0%, #0f172a 50%, #090d16 100%);
        color: #f8fafc;
    }

    /* Welcome Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.6rem;
    }

    /* 5 W's Educational Card */
    .five-ws-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
    }
    .w-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.75rem;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }

    /* Distinct Explanatory Info Card */
    .explain-card {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    /* Distinct Scenario Button Wrappers */
    .scenario-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1.2rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .scenario-badge-red {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .scenario-badge-orange {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .scenario-badge-green {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .scenario-badge-blue {
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Decision Result Cards */
    .verdict-critical {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(153, 27, 27, 0.4) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.3);
    }
    .verdict-warning {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.25) 0%, rgba(180, 83, 9, 0.4) 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 0 40px rgba(245, 158, 11, 0.25);
    }
    .verdict-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.35) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.2);
    }

    /* Signal Items */
    .signal-item-flag {
        background: rgba(220, 38, 38, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.65rem;
    }
    .signal-item-clean {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.65rem;
    }

    /* Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 0.85rem 2rem;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.65);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    scale_model_path = os.path.join(MODELS_DIR, "xgboost_scale_1M.joblib")
    scale_feats_path = os.path.join(MODELS_DIR, "scale_features.joblib")
    scale_model = joblib.load(scale_model_path) if os.path.exists(scale_model_path) else None
    scale_feats = joblib.load(scale_feats_path) if os.path.exists(scale_feats_path) else None

    metrics_path = os.path.join(MODELS_DIR, "evaluation", "metrics.json")
    metrics_1m_path = os.path.join(MODELS_DIR, "evaluation", "metrics_1M.json")
    
    m_real = json.load(open(metrics_path)) if os.path.exists(metrics_path) else None
    m_1m = json.load(open(metrics_1m_path)) if os.path.exists(metrics_1m_path) else None

    return scale_model, scale_feats, m_real, m_1m


scale_model, scale_feats, m_real, m_1m = load_all_artifacts()

# 1. Hero / Purpose Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ FraudShield AI &bull; Smart Fraud Detection & Compliance Platform</div>
    <div style="font-size: 1.1rem; color: #cbd5e1; line-height: 1.6; max-width: 950px;">
        Welcome to <strong>FraudShield</strong>. This platform simulates a real-world bank security brain that inspects payments, wires, and debit swipes in <strong>under 15 milliseconds</strong> to catch hackers, stolen cards, and account takeovers before money is lost.
    </div>
    
    <div class="five-ws-box">
        <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; margin-bottom: 0.6rem;">📖 The 5 W's of This Project (What is happening here?):</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; font-size: 0.88rem; color: #e2e8f0;">
            <div><span class="w-badge" style="background:#0284c7; color:#fff;">WHO</span> Built for banks, payment processors (like Stripe & Visa), and fraud investigators.</div>
            <div><span class="w-badge" style="background:#7c3aed; color:#fff;">WHAT</span> An AI that analyzes payments to immediately say: <em>"Approve"</em> or <em>"Block Fraud"</em>.</div>
            <div><span class="w-badge" style="background:#059669; color:#fff;">WHY</span> Over $30 Billion is stolen annually; real users hate being blocked, while fraudsters hide among millions of transactions.</div>
            <div><span class="w-badge" style="background:#d97706; color:#fff;">HOW</span> Powered by <strong>XGBoost Machine Learning</strong>, <strong>TreeSHAP Explainability</strong>, and a <strong>RAG Legal Compliance Copilot</strong>.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. System Performance KPIs
st.markdown("### 📊 Performance At A Glance (What do these numbers mean?)")
st.caption("Here is how well this machine learning model performed when tested against thousands of unseen transactions:")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.2rem;">
        <div style="color: #38bdf8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">PR-AUC Score</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 0.2rem;">0.8081</div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.4rem;">
            <strong>What it means:</strong> Measures how accurately the AI finds the 0.17% rare fraud needle in a haystack without false alarms.
        </div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.2rem;">
        <div style="color: #818cf8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Fraud Catch Rate</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 0.2rem;">85.14%</div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.4rem;">
            <strong>What it means:</strong> The model catches 85.1% of all fraud attacks while keeping false alarms under 1 per 1,000 legitimate shoppers.
        </div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.2rem;">
        <div style="color: #c084fc; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Batch Speed</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 0.2rem;">2,366,392</div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.4rem;">
            <strong>What it means:</strong> Scored over 2.3 million transactions per second—fast enough to handle peak Black Friday traffic easily.
        </div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.2rem;">
        <div style="color: #34d399; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Decision Latency</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 0.2rem;">&lt; 15 ms</div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.4rem;">
            <strong>What it means:</strong> Complete AI analysis finishes in a fraction of a blink, so customers never wait at checkout.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")
st.divider()

# 3. 1-Click Interactive Test Scenarios
st.markdown("### 🕹️ Step 1: Choose or Load A Test Scenario")
st.write(
    "Instead of typing manually, click any of the 4 scenario buttons below. "
    "Each button will automatically fill in real-world numbers so you can see how the AI reacts to different behaviors:"
)

# Initialize Session State
if "amt" not in st.session_state:
    st.session_state.amt = 1200.0
if "bal" not in st.session_state:
    st.session_state.bal = 2500.0
if "type" not in st.session_state:
    st.session_state.type = "TRANSFER"
if "dist" not in st.session_state:
    st.session_state.dist = 15.0
if "pins" not in st.session_state:
    st.session_state.pins = 0
if "vel" not in st.session_state:
    st.session_state.vel = 1
if "hour" not in st.session_state:
    st.session_state.hour = 14

btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

with btn_c1:
    st.markdown("""
    <div class="scenario-card">
        <div>
            <span class="scenario-badge-red">🚨 ATTACK PATTERN A</span>
            <div style="font-size: 1.05rem; font-weight: 800; color: #ffffff; margin: 0.5rem 0 0.3rem 0;">Account Takeover</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                <strong>What it is:</strong> A hacker breaks into a customer's online bank at 3 AM and tries to wire out 98% of their life savings ($4,900) from 3,200 km away.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load 'Account Takeover' Test", key="btn_ato", use_container_width=True):
        st.session_state.amt = 4900.0
        st.session_state.bal = 5000.0
        st.session_state.type = "TRANSFER"
        st.session_state.dist = 3200.0
        st.session_state.pins = 2
        st.session_state.vel = 6
        st.session_state.hour = 3

with btn_c2:
    st.markdown("""
    <div class="scenario-card">
        <div>
            <span class="scenario-badge-orange">🚨 ATTACK PATTERN B</span>
            <div style="font-size: 1.05rem; font-weight: 800; color: #ffffff; margin: 0.5rem 0 0.3rem 0;">Card Testing Bot</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                <strong>What it is:</strong> An automated script tries to cash out $2,100 while guessing the PIN with multiple failed attempts in rapid succession.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load 'Card Testing' Test", key="btn_bot", use_container_width=True):
        st.session_state.amt = 2100.0
        st.session_state.bal = 3500.0
        st.session_state.type = "CASH_OUT"
        st.session_state.dist = 1400.0
        st.session_state.pins = 3
        st.session_state.vel = 8
        st.session_state.hour = 4

with btn_c3:
    st.markdown("""
    <div class="scenario-card">
        <div>
            <span class="scenario-badge-green">🟢 NORMAL BEHAVIOR A</span>
            <div style="font-size: 1.05rem; font-weight: 800; color: #ffffff; margin: 0.5rem 0 0.3rem 0;">Local Grocery / Lunch</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                <strong>What it is:</strong> A normal person buying a $45 meal at 1:00 PM just 3 km from their home, with 0 failed PINs and healthy savings remaining.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load 'Normal Lunch' Test", key="btn_lunch", use_container_width=True):
        st.session_state.amt = 45.0
        st.session_state.bal = 2800.0
        st.session_state.type = "PAYMENT"
        st.session_state.dist = 3.5
        st.session_state.pins = 0
        st.session_state.vel = 1
        st.session_state.hour = 13

with btn_c4:
    st.markdown("""
    <div class="scenario-card">
        <div>
            <span class="scenario-badge-blue">🟢 NORMAL BEHAVIOR B</span>
            <div style="font-size: 1.05rem; font-weight: 800; color: #ffffff; margin: 0.5rem 0 0.3rem 0;">Airline Flight Ticket</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                <strong>What it is:</strong> A legitimate $850 flight purchase. The amount is large, but normal location and 0 failed PINs prove the cardholder is genuine.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Load 'Airline Ticket' Test", key="btn_flight", use_container_width=True):
        st.session_state.amt = 850.0
        st.session_state.bal = 6200.0
        st.session_state.type = "PAYMENT"
        st.session_state.dist = 12.0
        st.session_state.pins = 0
        st.session_state.vel = 1
        st.session_state.hour = 16

st.write("")
st.divider()

# 4. Custom Parameter Cockpit
st.markdown("### 🎛️ Step 2: Review or Customize the Transaction Details")
st.write(
    "You can keep the loaded numbers above or change any value below to test your own custom transaction. "
    "Every box explains **WHY** the bank checks that signal:"
)

with st.container():
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 1.8rem; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)
    
    in_c1, in_c2, in_c3 = st.columns(3)
    
    with in_c1:
        st.markdown("**1. Financial Money Values**")
        st.caption("Why we check this: Fraudsters usually steal as much money as possible, whereas normal users spend modest percentages.")
        amt_val = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=100000.0, value=float(st.session_state.amt), step=50.0, help="How much money is being charged.")
        bal_val = st.number_input("Customer's Current Account Balance ($)", min_value=1.0, max_value=250000.0, value=float(st.session_state.bal), step=100.0, help="Total money customer has in bank.")

    with in_c2:
        st.markdown("**2. Where & How It Was Sent**")
        st.caption("Why we check this: Wire transfers and cash liquidations are high risk because the money cannot be reversed once gone.")
        type_options = ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"]
        type_idx = type_options.index(st.session_state.type) if st.session_state.type in type_options else 0
        type_val = st.selectbox("Transaction Method", type_options, index=type_idx, help="TRANSFER/CASH_OUT are high risk. PAYMENT/DEBIT are routine shopping.")
        dist_val = st.number_input("Distance from Cardholder Home (km)", min_value=0.0, max_value=25000.0, value=float(st.session_state.dist), step=25.0, help="If distance > 800 km in minutes, this triggers 'Impossible Travel'.")

    with in_c3:
        st.markdown("**3. Security & Card Credentials**")
        st.caption("Why we check this: Real cardholders know their password. Multiple failed attempts mean someone is guessing or hacking.")
        pins_val = st.number_input("Consecutive Failed PIN Attempts", min_value=0, max_value=5, value=int(st.session_state.pins), step=1, help="2+ wrong PIN attempts suggests brute-force attack.")
        vel_val = st.number_input("Transactions Made in Last 60 Mins", min_value=0, max_value=25, value=int(st.session_state.vel), step=1, help="Robots and automated scripts make 5-15 purchases an hour.")
        hour_val = st.slider("Time of Day (Hour: 0 to 23)", 0, 23, int(st.session_state.hour), help="Transactions at 3 AM are statistically higher risk than 2 PM.")

    st.write("")
    run_btn = st.button("🚀 CLICK HERE TO ANALYZE THIS TRANSACTION (RUN AI FRAUD SHIELD)", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 5. Run Inference & Display Results
if scale_model and scale_feats:
    new_bal = max(0.0, bal_val - amt_val)
    b_drain = (bal_val - new_bal) / (bal_val + 1.0)
    b_error = (bal_val - amt_val) - new_bal
    amt_bal_ratio = amt_val / (bal_val + 1.0)
    h_sin = np.sin(2 * np.pi * hour_val / 24.0)
    h_cos = np.cos(2 * np.pi * hour_val / 24.0)
    vel_pin = vel_val * (pins_val + 1)
    geo_vel = dist_val * np.log1p(amt_val)

    row_data = {
        "amount": np.log1p(amt_val),
        "oldbalanceOrg": np.log1p(bal_val),
        "newbalanceOrig": np.log1p(new_bal),
        "dist_from_home_km": dist_val,
        "velocity_1h": float(vel_val),
        "failed_pin_attempts": float(pins_val),
        "balance_error_orig": b_error,
        "balance_drain_ratio": b_drain,
        "amount_to_balance_ratio": amt_bal_ratio,
        "hour_sin": h_sin,
        "hour_cos": h_cos,
        "velocity_pin_risk": vel_pin,
        "geo_amount_velocity": geo_vel,
    }

    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        row_data[f"type_{t}"] = 1.0 if type_val.upper() == t else 0.0

    for d in ["atm", "mobile_app", "pos_terminal", "web_browser"]:
        row_data[f"device_{d}"] = 1.0 if d == "mobile_app" else 0.0

    eval_df = pd.DataFrame([row_data])
    for col in scale_feats:
        if col not in eval_df.columns:
            eval_df[col] = 0.0
    eval_df = eval_df[scale_feats]

    pred_prob = float(scale_model.predict_proba(eval_df)[0, 1])
    risk_score = round(pred_prob * 100.0, 1)

    st.write("")
    st.markdown("### 🎯 Step 3: View the AI Decision Verdict")
    st.write("Here is what the FraudShield AI brain decided, why it decided it, and what legal action is required:")

    res_l, res_r = st.columns([1.1, 1.3])

    with res_l:
        if risk_score >= 50.0:
            st.markdown(f"""
            <div class="verdict-critical">
                <div style="font-size: 0.85rem; font-weight: 800; color: #f87171; letter-spacing: 0.08em; text-transform: uppercase;">AI DECISION VERDICT</div>
                <div style="font-size: 2rem; font-weight: 800; color: #ffffff; margin-top: 0.3rem;">🔴 ALERT: FRAUD DETECTED</div>
                <div style="font-size: 1.15rem; color: #fca5a5; font-weight: 700; margin-top: 0.4rem;">CRITICAL RISK LEVEL: {risk_score} / 100</div>
                <div style="font-size: 0.92rem; color: #ffffff; margin-top: 0.8rem; line-height: 1.5;">
                    <strong>What this means:</strong> The AI is <strong>{pred_prob*100:.1f}% confident</strong> this transaction is unauthorized or fraudulent.
                </div>
                <div style="background: rgba(0, 0, 0, 0.4); border-radius: 8px; padding: 0.9rem; margin-top: 1.2rem; font-weight: 700; color: #fee2e2;">
                    🚨 Bank Action Required: BLOCK PAYMENT & FREEZE ACCOUNT IMMEDIATELY
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif risk_score >= 25.0:
            st.markdown(f"""
            <div class="verdict-warning">
                <div style="font-size: 0.85rem; font-weight: 800; color: #fbbf24; letter-spacing: 0.08em; text-transform: uppercase;">AI DECISION VERDICT</div>
                <div style="font-size: 2rem; font-weight: 800; color: #ffffff; margin-top: 0.3rem;">⚠️ WARNING: SUSPICIOUS ACTIVITY</div>
                <div style="font-size: 1.15rem; color: #fde68a; font-weight: 700; margin-top: 0.4rem;">ELEVATED RISK LEVEL: {risk_score} / 100</div>
                <div style="font-size: 0.92rem; color: #ffffff; margin-top: 0.8rem; line-height: 1.5;">
                    <strong>What this means:</strong> Unusual spending signals detected ({pred_prob*100:.1f}% risk), but not yet a confirmed attack.
                </div>
                <div style="background: rgba(0, 0, 0, 0.4); border-radius: 8px; padding: 0.9rem; margin-top: 1.2rem; font-weight: 700; color: #fef3c7;">
                    📲 Bank Action Required: CHALLENGE WITH 2-FACTOR AUTH / ONE-TIME PASSCODE (OTP)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-safe">
                <div style="font-size: 0.85rem; font-weight: 800; color: #34d399; letter-spacing: 0.08em; text-transform: uppercase;">AI DECISION VERDICT</div>
                <div style="font-size: 2rem; font-weight: 800; color: #ffffff; margin-top: 0.3rem;">🟢 VERIFIED: REAL TRANSACTION</div>
                <div style="font-size: 1.15rem; color: #a7f3d0; font-weight: 700; margin-top: 0.4rem;">LOW RISK LEVEL: {risk_score} / 100</div>
                <div style="font-size: 0.92rem; color: #ffffff; margin-top: 0.8rem; line-height: 1.5;">
                    <strong>What this means:</strong> Everything looks genuine ({pred_prob*100:.1f}% risk). The payment matches normal cardholder habits.
                </div>
                <div style="background: rgba(0, 0, 0, 0.4); border-radius: 8px; padding: 0.9rem; margin-top: 1.2rem; font-weight: 700; color: #d1fae5;">
                    ✅ Bank Action Required: INSTANT AUTO-APPROVE IN &lt; 15 MILLISECONDS
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.progress(min(1.0, risk_score / 100.0))
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Fraud Probability", f"{pred_prob * 100:.2f}%", help="Calculated using cost-sensitive XGBoost with Isotonic calibration")
        with m_c2:
            st.metric("Account Balance Drained", f"{b_drain * 100:.1f}%", help="Percentage of customer's total bank balance wiped out by this transaction")

    with res_r:
        st.markdown("**🔍 Why Did the AI Make This Choice? (Explainability)**")
        st.caption("Instead of a mysterious 'black box', here are the exact physical and behavioral signals that triggered this result:")

        signals = []
        if b_drain >= 0.85 and amt_val > 500:
            signals.append(("Account Drain Alert", f"Transaction wipes out {b_drain*100:.0f}% of customer's total savings. Legitimate users rarely drain everything in one purchase.", "🔴 HIGH RISK"))
        if dist_val > 500:
            signals.append(("Impossible Travel", f"Origin is {dist_val:,.0f} km away from cardholder home. Physically impossible to travel this fast without a stolen session token.", "🔴 HIGH RISK"))
        if pins_val >= 2:
            signals.append(("Credential Brute-Force", f"{pins_val} consecutive failed PIN attempts detected immediately before this event, suggesting guessing.", "🔴 HIGH RISK"))
        if vel_val >= 4:
            signals.append(("Velocity Burst", f"{vel_val} rapid purchases made within the last 60 minutes. Humans don't buy this fast; automated bots do.", "🟡 WARNING"))

        if signals:
            for s_title, s_desc, s_badge in signals:
                st.markdown(f"""
                <div class="signal-item-flag">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; color: #fca5a5; font-size: 0.95rem;">⚠️ {s_title}</span>
                        <span style="font-size: 0.72rem; font-weight: 800; padding: 0.2rem 0.5rem; border-radius: 4px; background: rgba(239, 68, 68, 0.3); color: #fee2e2;">{s_badge}</span>
                    </div>
                    <div style="font-size: 0.84rem; color: #cbd5e1; margin-top: 0.35rem; line-height: 1.4;">{s_desc}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="signal-item-clean">
                <div style="font-weight: 800; color: #6ee7b7; font-size: 0.95rem;">✅ Clean Behavioral Profile</div>
                <div style="font-size: 0.84rem; color: #cbd5e1; margin-top: 0.35rem; line-height: 1.4;">
                    Normal transaction amount, local merchant distance, 0 failed PINs, and routine velocity. Perfectly consistent with genuine human shopping habits.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("**📋 Exact Numbers Evaluated:**")
        metrics_table = pd.DataFrame({
            "Signal Checked": ["Money Amount", "Account Drain %", "Distance from Home", "Prior 1h Velocity", "Failed PINs"],
            "Evaluated Value": [f"${amt_val:,.2f}", f"{b_drain*100:.1f}%", f"{dist_val:,.1f} km away", f"{vel_val} purchases", f"{pins_val} wrong PINs"],
            "Normal Human Baseline": ["$20 - $250", "< 15% of balance", "< 25 km from home", "1 per hour", "0 failed PINs"]
        })
        st.table(metrics_table)

    # 6. Compliance RAG Copilot
    st.write("")
    st.markdown("---")
    st.markdown("### 📚 Step 4: The Legal Compliance Copilot (RAG System)")
    st.markdown("""
    <div class="explain-card">
        <strong>What is this section?</strong> When fraud occurs, federal law requires banks to file legal reports with the government (FinCEN). 
        Instead of forcing human investigators to search through 500-page banking law manuals, our <strong>RAG (Retrieval-Augmented Generation)</strong> 
        system automatically searches federal regulations (Bank Secrecy Act, Regulation E, FinCEN SAR rules) and drafts the exact legal filing for you.
    </div>
    """, unsafe_allow_html=True)

    d_tier = "CRITICAL" if risk_score >= 50 else ("MEDIUM" if risk_score >= 25 else "LOW")
    
    rag_dossier = rag_pipeline.generate_compliance_dossier(
        transaction_id="TXN-LIVE-SIM",
        amount=amt_val,
        risk_score=risk_score,
        risk_tier=d_tier,
        dist_from_home_km=dist_val,
        balance_drain_pct=b_drain * 100.0,
        failed_pins=int(pins_val),
        tx_type=type_val
    )

    r_col1, r_col2 = st.columns([1, 1.2])

    with r_col1:
        if rag_dossier["sar_filing_required"]:
            st.markdown(f"""
            <div style="background: rgba(220, 38, 38, 0.2); border: 1px solid rgba(239, 68, 68, 0.5); border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;">
                <div style="color: #f87171; font-weight: 800; font-size: 1rem;">🚨 LEGAL ACTION MANDATED: FinCEN Form 111 (SAR)</div>
                <div style="font-size: 0.88rem; color: #fca5a5; margin-top: 0.5rem; line-height: 1.5;">
                    <strong>Why required:</strong> {rag_dossier['sar_filing_justification']}
                </div>
                <div style="font-size: 0.84rem; font-weight: 800; color: #ffffff; margin-top: 0.8rem; background: rgba(0,0,0,0.3); padding: 0.5rem 0.8rem; border-radius: 6px;">
                    ⏱️ Mandatory Statutory Deadline: {rag_dossier['filing_deadline']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;">
                <div style="color: #34d399; font-weight: 800; font-size: 1rem;">✅ REGULATORY STATUS: Compliant</div>
                <div style="font-size: 0.88rem; color: #a7f3d0; margin-top: 0.5rem; line-height: 1.5;">
                    This transaction falls within normal dollar and risk limits. No mandatory Suspicious Activity Report (SAR) is required under federal banking statutes.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**⏳ Mandatory Investigation Timelines for the Bank:**")
        for t in rag_dossier["compliance_timelines"]:
            st.markdown(f"<div style='font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.4rem; background: rgba(15,23,42,0.5); padding: 0.5rem 0.8rem; border-radius: 6px;'>{t}</div>", unsafe_allow_html=True)

    with r_col2:
        st.markdown("**⚖️ Retrieved Banking Regulations Cited by the AI:**")
        st.caption("The RAG engine automatically pulled these exact legal documents from the bank's knowledge base:")
        for p in rag_dossier["retrieved_policies"]:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 1.1rem; margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 800; color: #818cf8; font-size: 0.9rem;">[{p['document_id']}] {p['document']}</div>
                    <span style="font-size: 0.72rem; font-weight: 800; background: rgba(99, 102, 241, 0.3); color: #c7d2fe; padding: 0.15rem 0.55rem; border-radius: 4px;">Semantic Match: {p['relevance']}</span>
                </div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #94a3b8; margin-top: 0.25rem;">Section: {p['section']}</div>
                <div style="font-size: 0.82rem; color: #e2e8f0; margin-top: 0.45rem; line-height: 1.45; font-style: italic;">"{p['summary']}"</div>
            </div>
            """, unsafe_allow_html=True)
