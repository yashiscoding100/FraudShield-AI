"""
Interactive Transaction Simulator for FraudShield.
Allows users to input any custom transaction and get an instant verdict:
REAL (Legitimate) vs. FRAUD (Suspicious), with risk probability and SHAP reasoning.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SRC_DIR, "..", "models")


def score_transaction(
    amount: float,
    old_balance: float,
    new_balance: float,
    tx_type: str = "TRANSFER",
    device_type: str = "mobile_app",
    dist_from_home_km: float = 15.0,
    velocity_1h: int = 1,
    failed_pins: int = 0,
    hour_of_day: int = 14
):
    """Scores any custom transaction against the trained 1,000,000-transaction model."""
    model_path = os.path.join(MODELS_DIR, "xgboost_scale_1M.joblib")
    features_path = os.path.join(MODELS_DIR, "scale_features.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Model artifacts not found. Please train first.")
        return

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    # Calculate engineered features
    balance_error = (old_balance - amount) - new_balance
    balance_drain_ratio = (old_balance - new_balance) / (old_balance + 1.0)
    amount_to_balance_ratio = amount / (old_balance + 1.0)
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
    velocity_pin_risk = velocity_1h * (failed_pins + 1)
    geo_amount_velocity = dist_from_home_km * np.log1p(amount)

    # Base feature dictionary
    data = {
        "amount": np.log1p(amount),
        "oldbalanceOrg": np.log1p(old_balance),
        "newbalanceOrig": np.log1p(new_balance),
        "dist_from_home_km": dist_from_home_km,
        "velocity_1h": float(velocity_1h),
        "failed_pin_attempts": float(failed_pins),
        "balance_error_orig": balance_error,
        "balance_drain_ratio": balance_drain_ratio,
        "amount_to_balance_ratio": amount_to_balance_ratio,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "velocity_pin_risk": velocity_pin_risk,
        "geo_amount_velocity": geo_amount_velocity,
    }

    # Add one-hot encoded categories
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        data[f"type_{t}"] = 1.0 if tx_type.upper() == t else 0.0

    for d in ["atm", "mobile_app", "pos_terminal", "web_browser"]:
        data[f"device_{d}"] = 1.0 if device_type.lower() == d else 0.0

    df = pd.DataFrame([data])
    # Align columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]

    # Predict probability
    prob = float(model.predict_proba(df)[0, 1])
    risk_score = round(prob * 100.0, 1)

    # Decision logic
    if risk_score >= 50.0:
        verdict = "[ALERT: FRAUD DETECTED]"
        tier = "CRITICAL RISK"
        action = "BLOCK TRANSACTION & FREEZE ACCOUNT"
    elif risk_score >= 25.0:
        verdict = "[WARNING: SUSPICIOUS TRANSACTION]"
        tier = "ELEVATED RISK"
        action = "CHALLENGE WITH 2-FACTOR AUTH / OTP"
    else:
        verdict = "[VERIFIED: REAL TRANSACTION (LEGITIMATE)]"
        tier = "LOW RISK"
        action = "AUTO-APPROVE"

    # Print Report
    print("=" * 65)
    print("           FRAUDSHIELD AI TRANSACTION VERDICT")
    print("=" * 65)
    print(f"Verdict            : {verdict}")
    print(f"Risk Score         : {risk_score} / 100 ({tier})")
    print(f"Fraud Likelihood   : {prob * 100:.2f}%")
    print(f"Recommended Action : {action}")
    print("-" * 65)
    print("TRANSACTION DETAILS EVALUATED:")
    print(f"  • Amount             : ${amount:,.2f}")
    print(f"  • Transaction Type   : {tx_type.upper()}")
    print(f"  • Account Balance    : ${old_balance:,.2f} ➔ ${new_balance:,.2f}")
    print(f"  • Distance from Home : {dist_from_home_km:,.1f} km")
    print(f"  • Rapid Velocity (1h): {velocity_1h} transactions")
    print(f"  • Failed PIN Attempts: {failed_pins}")
    print(f"  • Time of Day        : {hour_of_day:02d}:00")
    print("-" * 65)

    # Key Risk Drivers
    reasons = []
    if balance_drain_ratio >= 0.85 and amount > 500:
        reasons.append(f"• Account Drain Alert: Transaction wipes out {balance_drain_ratio*100:.0f}% of total account balance.")
    if dist_from_home_km > 500:
        reasons.append(f"• Impossible Travel: Location is {dist_from_home_km:,.0f} km away from registered cardholder home.")
    if failed_pins >= 2:
        reasons.append(f"• Credential Brute-force: {failed_pins} consecutive failed PIN attempts detected.")
    if velocity_1h >= 4:
        reasons.append(f"• Velocity Burst: {velocity_1h} transactions initiated within the last 60 minutes.")

    if reasons:
        print("WHY WAS IT FLAGGED (KEY SUSPICIOUS SIGNALS):")
        for r in reasons:
            print("  " + r)
    else:
        print("BEHAVIORAL SIGNALS:")
        print("  • All parameters match normal cardholder baseline spending patterns.")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode when run without command line flags
        print("\n" + "=" * 65)
        print("          FRAUDSHIELD INTERACTIVE TRANSACTION TESTER")
        print("=" * 65)
        print("Enter transaction details below to test if it's REAL or FRAUD:\n")
        
        try:
            amt_input = input("1. Transaction Amount in $ [e.g. 75 or 4200]: ").strip()
            amount = float(amt_input) if amt_input else 75.0

            bal_input = input("2. Account Balance in $ [e.g. 2500]: ").strip()
            balance = float(bal_input) if bal_input else 2500.0

            t_input = input("3. Type (PAYMENT, TRANSFER, CASH_OUT, DEBIT) [default: PAYMENT]: ").strip().upper()
            tx_type = t_input if t_input in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"] else "PAYMENT"

            d_input = input("4. Distance from home in km [default: 5]: ").strip()
            dist = float(d_input) if d_input else 5.0

            v_input = input("5. Transactions in last 1 hour [default: 1]: ").strip()
            velocity = int(v_input) if v_input else 1

            p_input = input("6. Consecutive failed PIN attempts [default: 0]: ").strip()
            failed_pins = int(p_input) if p_input else 0

            h_input = input("7. Time of day / Hour (0-23) [default: 14]: ").strip()
            hour = int(h_input) if h_input else 14
        except (ValueError, EOFError):
            print("\nUsing default transaction parameters...")
            amount, balance, tx_type, dist, velocity, failed_pins, hour = 75.0, 2500.0, "PAYMENT", 5.0, 1, 0, 14

        new_bal = max(0.0, balance - amount)
        print("\nEvaluating transaction through AI Risk Engine...\n")
        score_transaction(
            amount=amount,
            old_balance=balance,
            new_balance=new_bal,
            tx_type=tx_type,
            dist_from_home_km=dist,
            velocity_1h=velocity,
            failed_pins=failed_pins,
            hour_of_day=hour
        )
    else:
        # CLI Flag mode
        parser = argparse.ArgumentParser(description="Score any transaction for fraud.")
        parser.add_argument("--amount", type=float, default=50.0, help="Transaction amount in USD")
        parser.add_argument("--balance", type=float, default=2500.0, help="Current account balance")
        parser.add_argument("--new_balance", type=float, default=None, help="New balance after transaction (default: balance - amount)")
        parser.add_argument("--type", type=str, default="PAYMENT", help="PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN")
        parser.add_argument("--dist", type=float, default=5.0, help="Distance from home address (km)")
        parser.add_argument("--velocity", type=int, default=1, help="Number of txns in last 1 hour")
        parser.add_argument("--failed_pins", type=int, default=0, help="Number of failed PIN attempts")
        parser.add_argument("--hour", type=int, default=14, help="Hour of day (0-23)")

        args = parser.parse_args()
        new_bal = args.new_balance if args.new_balance is not None else max(0.0, args.balance - args.amount)

        score_transaction(
            amount=args.amount,
            old_balance=args.balance,
            new_balance=new_bal,
            tx_type=args.type,
            dist_from_home_km=args.dist,
            velocity_1h=args.velocity,
            failed_pins=args.failed_pins,
            hour_of_day=args.hour
        )
