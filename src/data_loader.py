"""
Data acquisition, validation, and stratified splitting for the Credit Card Fraud benchmark.
"""

import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FraudShield.DataLoader")

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "creditcard.pkl")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def download_or_load_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Downloads the European Credit Card Fraud Detection benchmark (OpenML ID: 1597)
    or loads it from local cache if already downloaded.
    
    Contains 284,807 transactions with 492 frauds (0.172% fraud rate).
    """
    raw_dir = os.path.dirname(RAW_DATA_PATH)
    os.makedirs(raw_dir, exist_ok=True)

    if os.path.exists(RAW_DATA_PATH) and not force_download:
        logger.info(f"Loading cached dataset from {RAW_DATA_PATH}...")
        df = pd.read_pickle(RAW_DATA_PATH)
        logger.info(f"Loaded {len(df):,} transactions successfully.")
        return df

    logger.info("Fetching OpenML Credit Card Fraud Detection benchmark (data_id=1597)...")
    try:
        dataset = fetch_openml(data_id=1597, as_frame=True, parser="auto")
        df = dataset.frame
    except Exception as e:
        logger.warning(f"OpenML fetch by ID failed ({e}), trying by name 'CreditCardFraudDetection'...")
        dataset = fetch_openml("CreditCardFraudDetection", version=1, as_frame=True, parser="auto")
        df = dataset.frame

    # Normalize column names & data types
    df.columns = [str(c).strip() for c in df.columns]
    
    # Target column might be 'Class' or string '0'/'1'
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)
    else:
        raise ValueError("Target column 'Class' not found in fetched dataset.")

    # Convert numeric fields
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any nulls if present
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing values.")

    # Save to pickle for fast, dependency-free local I/O
    df.to_pickle(RAW_DATA_PATH)
    logger.info(f"Saved dataset with {len(df):,} rows and {len(df.columns)} columns to {RAW_DATA_PATH}.")
    
    fraud_count = int(df["Class"].sum())
    fraud_rate = (fraud_count / len(df)) * 100
    logger.info(f"Class distribution: Legitimate={len(df)-fraud_count:,}, Fraud={fraud_count:,} ({fraud_rate:.3f}%)")

    return df


def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.15, 
    val_size: float = 0.15, 
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Performs stratified train / validation / test splits.
    Stratification is crucial due to the extreme class imbalance (0.172% fraud).
    
    Returns:
        Dict containing train, val, and test DataFrames.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Split into train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Class"],
        random_state=random_state
    )
    
    # 2. Split train+val into train and val
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["Class"],
        random_state=random_state
    )

    logger.info(f"Split sizes - Train: {len(train_df):,} ({train_df['Class'].sum()} frauds), "
                f"Val: {len(val_df):,} ({val_df['Class'].sum()} frauds), "
                f"Test: {len(test_df):,} ({test_df['Class'].sum()} frauds)")

    # Save splits
    train_df.to_pickle(os.path.join(PROCESSED_DIR, "train.pkl"))
    val_df.to_pickle(os.path.join(PROCESSED_DIR, "val.pkl"))
    test_df.to_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }


if __name__ == "__main__":
    df = download_or_load_dataset()
    splits = split_data(df)
    print("Data download and split completed successfully.")
