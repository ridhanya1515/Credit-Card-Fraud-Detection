"""
Utility functions for Credit Card Fraud Detection project.

This module provides helper functions for:
- Loading real Kaggle dataset
- Data validation
- Logging
"""

import numpy as np
import pandas as pd
import os
from src.config import (
    KAGGLE_DATASET_PATH, KAGGLE_SAMPLE_PATH, V_FEATURES, RANDOM_SEED
)


def load_kaggle_data(filepath=KAGGLE_DATASET_PATH):
    """
    Load real Kaggle Credit Card Fraud Detection dataset.
    
    Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    Expected format:
    - V1 to V28: PCA-transformed features
    - Amount: Transaction amount
    - Time: Time in seconds since first transaction
    - Class: 0 (legitimate), 1 (fraud)
    
    Returns:
        pd.DataFrame: Dataset with all features and Class column
    
    Raises:
        FileNotFoundError: If dataset not found at expected location
    """
    if not os.path.exists(filepath):
        if os.path.exists(KAGGLE_SAMPLE_PATH):
            filepath = KAGGLE_SAMPLE_PATH
        else:
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"❌ Dataset not found at: {filepath}\n\n"
                f"To use this project, you need the Kaggle Credit Card Fraud dataset:\n"
                f"1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                f"2. Download creditcard.csv\n"
                f"3. Place it in: data/creditcard.csv\n"
                f"{'='*70}\n"
            )
    
    df = pd.read_csv(filepath)
    
    # Validate dataset structure
    required_cols = V_FEATURES + ["Amount", "Time", "Class"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    # Check for missing values in class column
    if df["Class"].isnull().any():
        df = df.dropna(subset=["Class"])
    
    return df


def print_dataset_info(df):
    """
    Print summary information about the dataset.
    
    Parameters:
        df (pd.DataFrame): Input dataset with 'Class' column
    """
    total = len(df)
    legit = (df["Class"] == 0).sum()
    fraud = (df["Class"] == 1).sum()
    fraud_pct = (fraud / total) * 100
    
    print(f"   ✓ Total Transactions: {total:,}")
    print(f"   ✓ Legitimate: {legit:,} ({(legit/total)*100:.2f}%)")
    print(f"   ✓ Fraudulent: {fraud:,} ({fraud_pct:.2f}%)")


def print_class_balance(y):
    """
    Print class balance information.
    
    Parameters:
        y (np.ndarray or pd.Series): Binary labels
    """
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = (count / len(y)) * 100
        label_name = "Fraudulent" if label == 1 else "Legitimate"
        print(f"   ✓ {label_name}: {count:,} ({pct:.2f}%)")
