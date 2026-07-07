"""
Data preprocessing module for Credit Card Fraud Detection.

This module handles:
- Feature scaling (with NO data leakage)
- Missing value handling
- Class imbalance using SMOTE and undersampling
- Train/test split with stratification

IMPORTANT: Train/test split is done BEFORE any preprocessing to avoid data leakage.
The scaler is fit ONLY on training data and applied to test data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import (
    RANDOM_SEED, TEST_SIZE, V_FEATURES, SCALED_FEATURES,
    SMOTE_SAMPLING_STRATEGY, UNDERSAMPLER_SAMPLING_STRATEGY
)


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Split data into train and test sets with stratification.
    
    **This is done BEFORE preprocessing to avoid data leakage.**
    
    Parameters:
        X (np.ndarray or pd.DataFrame): Features
        y (np.ndarray or pd.Series): Target labels
        test_size (float): Proportion of data for test set (default: 0.2)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class distribution
    )
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, feature_names=None):
    """
    Scale features using StandardScaler.
    
    **The scaler is fit ONLY on training data to prevent data leakage.**
    
    Parameters:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        feature_names (list): Names of features for DataFrame (optional)
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit scaler on training data ONLY
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply fitted scaler to test data
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame if feature names provided
    if feature_names is not None:
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train, random_state=RANDOM_SEED):
    """
    Handle class imbalance using SMOTE + RandomUnderSampler.
    
    **This is applied ONLY on training data.**
    
    SMOTE oversamples minority class (fraud), then RandomUnderSampler
    undersamples majority class (legitimate) to create balanced training data.
    
    Parameters:
        X_train (np.ndarray or pd.DataFrame): Training features
        y_train (np.ndarray or pd.Series): Training labels
        random_state (int): Random seed
    
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    # Step 1: SMOTE - Oversample minority class to 15% of majority
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=random_state
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Step 2: Random Undersampler - Undersample majority class to 50% of minority
    undersampler = RandomUnderSampler(
        sampling_strategy=UNDERSAMPLER_SAMPLING_STRATEGY,
        random_state=random_state
    )
    X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled


def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline with NO data leakage.
    
    Steps:
    1. Extract features and target
    2. Split data into train/test (BEFORE scaling to avoid leakage)
    3. Scale only Amount and Time (fit on training data only)
    4. Handle class imbalance (on training data only)
    
    Parameters:
        df (pd.DataFrame): Raw dataset with 'Class' column
    
    Returns:
        dict: Dictionary containing:
            - X_train_res: Resampled training features (scaled)
            - X_test: Test features (scaled)
            - y_train_res: Resampled training labels
            - y_test: Test labels
            - scaler: Fitted StandardScaler object
            - feature_columns: List of feature names used
    """
    # Extract features and target
    feature_columns = V_FEATURES + ["Amount", "Time"]
    X = df[feature_columns].values
    y = df["Class"].values
    
    # Step 1: Split data BEFORE any preprocessing
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 2: Scale ONLY Amount and Time (indices 28, 29)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit on training data ONLY
    X_train_scaled[:, [28, 29]] = scaler.fit_transform(X_train[:, [28, 29]])
    # Apply to test data
    X_test_scaled[:, [28, 29]] = scaler.transform(X_test[:, [28, 29]])
    
    # Step 3: Handle class imbalance (on training data only)
    X_train_res, y_train_res = handle_imbalance(X_train_scaled, y_train)
    
    return {
        "X_train_res": X_train_res,
        "X_test": X_test_scaled,
        "y_train_res": y_train_res,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": feature_columns
    }
