"""
Configuration file for Credit Card Fraud Detection project.

This module contains all configurable parameters for the project including:
- Dataset parameters
- Model hyperparameters
- File paths
- Feature names
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA PARAMETERS
# ============================================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2
KAGGLE_DATASET_PATH = os.path.join(DATA_DIR, "creditcard.csv")
# Smaller sample for GitHub/Streamlit Cloud (full dataset exceeds GitHub 100MB limit)
KAGGLE_SAMPLE_PATH = os.path.join(DATA_DIR, "creditcard_sample.csv")

# Real Kaggle dataset characteristics
# Actual values: 284,807 transactions, 492 fraud cases (0.172%)
N_FEATURES = 28  # PCA-transformed features (V1-V28)

# ============================================================================
# FEATURE NAMES
# ============================================================================
V_FEATURES = [f"V{i}" for i in range(1, N_FEATURES + 1)]
SPECIAL_FEATURES = ["Amount", "Time"]
SCALED_FEATURES = ["Amount_Scaled", "Time_Scaled"]
ALL_FEATURES = V_FEATURES + SPECIAL_FEATURES
FEATURE_COLS_PROCESSED = V_FEATURES + SCALED_FEATURES

TARGET_COLUMN = "Class"

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
SMOTE_SAMPLING_STRATEGY = 0.15  # Target 15% fraud after SMOTE
UNDERSAMPLER_SAMPLING_STRATEGY = 0.5  # Target 50% fraud after undersampling

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
MODEL_PARAMS = {
    "Logistic Regression": {
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "C": 0.1,
        "class_weight": "balanced",
    },
    "Random Forest": {
        "n_estimators": 100,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
        "n_jobs": -1,
        "max_depth": 15,
    },
    "Gradient Boosting": {
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 4,
        "subsample": 0.8,
        "random_state": RANDOM_SEED,
    },
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
THRESHOLD_RANGE = (0.2, 0.8)
THRESHOLD_STEP = 0.01

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
FIGURE_DPI = 150
FIGURE_STYLE = "seaborn-v0_8-darkgrid"
COLOR_LEGIT = "#2E75B6"
COLOR_FRAUD = "#C00000"
COLOR_TITLE = "#1F4E79"

# ============================================================================
# FILE NAMES
# ============================================================================
BEST_MODEL_NAME = "fraud_detection_model.joblib"
SCALER_NAME = "feature_scaler.joblib"
EDA_PLOT_NAME = "eda_analysis.png"
EVALUATION_PLOT_NAME = "model_evaluation.png"
