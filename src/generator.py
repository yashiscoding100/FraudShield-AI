"""
High-Throughput Vectorized Financial Stream Generator.
Generates 1,000,000+ realistic transactional events matching the PaySim / Sparkov 
FinTech architecture with realistic balance dynamics, velocity spikes, and fraud patterns.
"""

import os
import time
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.Generator")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def generate_million_stream(
    n_transactions: int = 1_000_000, 
    fraud_rate: float = 0.0015,  # 0.15% realistic industry fraud rate
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a 1,000,000+ transaction stream using vectorized NumPy.
    Runs in < 5 seconds and requires zero network downloads.
    """
    logger.info(f"Generating {n_transactions:,} transactions with {fraud_rate*100:.2f}% target fraud rate...")
    start_time = time.time()
    rng = np.random.RandomState(random_state)

    # 1. Temporal Dynamics: 720 hours (30-day simulation)
    step = rng.randint(1, 721, size=n_transactions, dtype=np.int32)
    hour_of_day = (step % 24).astype(np.int8)

    # 2. Transaction Types (matching PaySim distribution)
    type_choices = np.array(["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"])
    type_probs = np.array([0.35, 0.35, 0.20, 0.08, 0.02])
    tx_type = rng.choice(type_choices, size=n_transactions, p=type_probs)

    # 3. Base Amount Distribution (Log-normal: median ~$50, long tail up to $50k)
    base_amount = np.exp(rng.normal(loc=3.8, scale=1.3, size=n_transactions)).astype(np.float32)
    amount = np.round(np.clip(base_amount, 1.0, 50000.0), 2)

    # 4. Origin Account Balances
    old_balance_org = np.round(np.exp(rng.normal(loc=7.5, scale=1.5, size=n_transactions)), 2).astype(np.float32)
    new_balance_orig = np.maximum(0.0, old_balance_org - amount).astype(np.float32)

    # 5. Destination Account Balances
    old_balance_dest = np.round(np.exp(rng.normal(loc=6.8, scale=1.6, size=n_transactions)), 2).astype(np.float32)
    new_balance_dest = (old_balance_dest + amount).astype(np.float32)

    # 6. Channel, Distance & Velocity Features
    device_choices = np.array(["mobile_app", "web_browser", "pos_terminal", "atm"])
    device_type = rng.choice(device_choices, size=n_transactions, p=[0.55, 0.25, 0.15, 0.05])
    
    # Distance from cardholder home address (km)
    dist_from_home = np.round(rng.exponential(scale=12.0, size=n_transactions), 1).astype(np.float32)
    
    # Velocity: transactions initiated in the past 1 hour
    velocity_1h = rng.poisson(lam=0.4, size=n_transactions).astype(np.int8)
    failed_pin_attempts = rng.choice([0, 1, 2, 3], size=n_transactions, p=[0.97, 0.025, 0.004, 0.001]).astype(np.int8)

    # 7. Ground Truth Fraud Injection (Attack Scenarios)
    n_frauds = int(n_transactions * fraud_rate)
    fraud_indices = rng.choice(n_transactions, size=n_frauds, replace=False)
    is_fraud = np.zeros(n_transactions, dtype=np.int8)
    is_fraud[fraud_indices] = 1

    # Inject realistic fraud patterns into the positive samples:
    # Pattern A: Account Takeover & Account Drain (high TRANSFER/CASH_OUT emptying balance)
    half_f = n_frauds // 2
    f_ato = fraud_indices[:half_f]
    tx_type[f_ato] = rng.choice(["TRANSFER", "CASH_OUT"], size=len(f_ato), p=[0.6, 0.4])
    # Massive drain: amount equal to entire balance
    amount[f_ato] = np.round(old_balance_org[f_ato] * rng.uniform(0.90, 1.0, size=len(f_ato)), 2)
    new_balance_orig[f_ato] = np.maximum(0.0, old_balance_org[f_ato] - amount[f_ato])
    dist_from_home[f_ato] = np.round(rng.uniform(800.0, 4500.0, size=len(f_ato)), 1)
    velocity_1h[f_ato] = rng.randint(3, 9, size=len(f_ato), dtype=np.int8)

    # Pattern B: Card Testing / Brute Force
    f_test = fraud_indices[half_f:]
    failed_pin_attempts[f_test] = rng.choice([2, 3], size=len(f_test), p=[0.4, 0.6])
    velocity_1h[f_test] = rng.randint(4, 12, size=len(f_test), dtype=np.int8)
    dist_from_home[f_test] = np.round(rng.uniform(300.0, 2000.0, size=len(f_test)), 1)

    # Assemble DataFrame
    df = pd.DataFrame({
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrig": new_balance_orig,
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "device_type": device_type,
        "dist_from_home_km": dist_from_home,
        "velocity_1h": velocity_1h,
        "failed_pin_attempts": failed_pin_attempts,
        "hour_of_day": hour_of_day,
        "isFraud": is_fraud
    })

    elapsed = time.time() - start_time
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Generated {len(df):,} transactions ({df['isFraud'].sum():,} frauds, {mem_mb:.1f} MB) in {elapsed:.2f}s!")

    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_million_stream(1_000_000)
    out_path = os.path.join(DATA_DIR, "million_stream.pkl")
    df.to_pickle(out_path)
    logger.info(f"Stream saved to {out_path}")
