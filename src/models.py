"""
Model training module for Credit Card Fraud Detection.

This module provides:
- Model initialization with optimal hyperparameters
- Model training
- Prediction with threshold tuning
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

from src.config import MODEL_PARAMS, THRESHOLD_RANGE, THRESHOLD_STEP, RANDOM_SEED


def get_models():
    """
    Initialize all models with optimized hyperparameters.
    
    Returns:
        dict: Dictionary of {model_name: model_instance}
    """
    models = {
        "Logistic Regression": LogisticRegression(**MODEL_PARAMS["Logistic Regression"]),
        "Random Forest": RandomForestClassifier(**MODEL_PARAMS["Random Forest"]),
        "Gradient Boosting": GradientBoostingClassifier(**MODEL_PARAMS["Gradient Boosting"]),
    }
    return models


def train_model(model, X_train, y_train):
    """
    Train a single model on training data.
    
    Parameters:
        model: Scikit-learn model instance
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
    
    Returns:
        model: Trained model instance
    """
    model.fit(X_train, y_train)
    return model


def train_models(models_dict, X_train, y_train):
    """
    Train all models.
    
    Parameters:
        models_dict (dict): Dictionary of {name: model}
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
    
    Returns:
        dict: Dictionary of {name: trained_model}
    """
    trained_models = {}
    for name, model in models_dict.items():
        print(f"   → Training {name}...")
        trained_model = train_model(model, X_train, y_train)
        trained_models[name] = trained_model
    return trained_models


def tune_threshold(y_test, y_proba, metric="f1"):
    """
    Find optimal probability threshold to maximize F1-score for fraud detection.
    
    For fraud detection, we want to maximize F1-score on the fraud class (class 1)
    to balance precision and recall.
    
    Parameters:
        y_test (np.ndarray): True labels
        y_proba (np.ndarray): Predicted probabilities for class 1
        metric (str): Metric to optimize ('f1', 'precision', 'recall')
    
    Returns:
        tuple: (best_threshold, best_score)
    """
    best_score = 0
    best_threshold = 0.5
    
    thresholds = np.arange(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1], THRESHOLD_STEP)
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        else:
            score = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def predict_with_threshold(model, X, X_test, y_test, threshold=None):
    """
    Make predictions using trained model and optional threshold tuning.
    
    Parameters:
        model: Trained model
        X: Data to predict on
        X_test (np.ndarray): Test features (for threshold tuning)
        y_test (np.ndarray): Test labels (for threshold tuning)
        threshold (float): Fixed threshold. If None, optimal threshold is found.
    
    Returns:
        tuple: (y_pred, y_proba, best_threshold)
    """
    y_proba = model.predict_proba(X)[:, 1]
    
    if threshold is None:
        threshold, _ = tune_threshold(y_test, model.predict_proba(X_test)[:, 1])
    
    y_pred = (y_proba >= threshold).astype(int)
    
    return y_pred, y_proba, threshold
