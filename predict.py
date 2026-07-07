"""
Prediction module for Credit Card Fraud Detection.

This module provides functionality to:
- Load trained model and scaler from disk
- Preprocess new transaction data
- Make predictions on single or batch transactions
- Provide confidence scores

Usage:
    from predict import FraudDetectionPredictor
    
    predictor = FraudDetectionPredictor()
    prediction = predictor.predict(transaction_features)
    print(f"Fraud: {prediction['is_fraud']}, Confidence: {prediction['confidence']:.2%}")
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, BEST_MODEL_NAME, SCALER_NAME


class FraudDetectionPredictor:
    """Load and use trained fraud detection model for predictions."""
    
    def __init__(self, model_dir=MODELS_DIR):
        """
        Initialize predictor with saved model and scaler.
        
        Parameters:
            model_dir (str): Directory containing saved model files
        
        Raises:
            FileNotFoundError: If model or scaler files not found
        """
        self.model_path = os.path.join(model_dir, BEST_MODEL_NAME)
        self.scaler_path = os.path.join(model_dir, SCALER_NAME)
        self.features_path = os.path.join(model_dir, "feature_columns.joblib")
        
        # Check that files exist
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run train.py first.")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Run train.py first.")
        
        # Load model and scaler
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Load feature columns if available
        if os.path.exists(self.features_path):
            self.feature_columns = joblib.load(self.features_path)
        else:
            # Fallback: Generate feature names
            self.feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount_Scaled", "Time_Scaled"]
    
    def predict(self, features, return_probability=True):
        """
        Make prediction on transaction features.
        
        Parameters:
            features (array-like): Transaction features. Should be shape (30,) or (1, 30)
                                  corresponding to: V1-V28, Amount, Time
            return_probability (bool): Whether to return probability scores
        
        Returns:
            dict: Prediction results containing:
                - 'is_fraud': Boolean, True if transaction is flagged as fraud
                - 'confidence': Float, probability of fraud (0-1)
                - 'prediction': Int, 0 (legitimate) or 1 (fraud)
        
        Raises:
            ValueError: If features have incorrect shape
        """
        # Convert to numpy array and ensure correct shape
        features = np.asarray(features).reshape(1, -1)
        
        if features.shape[1] != 30:
            raise ValueError(
                f"Expected 30 features (V1-V28, Amount, Time), got {features.shape[1]}"
            )
        
        # Scale only Amount and Time (indices 28, 29)
        features_copy = features.copy().astype(float)
        features_copy[:, [28, 29]] = self.scaler.transform(features_copy[:, [28, 29]])
        
        # Make prediction
        probability = self.model.predict_proba(features_copy)[0, 1]
        prediction = self.model.predict(features_copy)[0]
        
        return {
            "is_fraud": bool(prediction == 1),
            "prediction": int(prediction),
            "confidence": float(probability)
        }
    
    def predict_batch(self, features):
        """
        Make predictions on multiple transactions.
        
        Parameters:
            features (array-like): Batch of transaction features.
                                  Shape: (n_samples, 30)
        
        Returns:
            list: List of prediction dictionaries
        """
        features = np.asarray(features)
        
        if features.shape[1] != 30:
            raise ValueError(
                f"Expected 30 features (V1-V28, Amount, Time), got {features.shape[1]}"
            )
        
        # Scale only Amount and Time columns
        features_copy = features.copy().astype(float)
        features_copy[:, [28, 29]] = self.scaler.transform(features_copy[:, [28, 29]])
        
        # Make predictions
        predictions = self.model.predict(features_copy)
        probabilities = self.model.predict_proba(features_copy)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "is_fraud": bool(pred == 1),
                "prediction": int(pred),
                "confidence": float(prob)
            })
        
        return results
    
    def predict_dataframe(self, df):
        """
        Make predictions on a DataFrame of transactions.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing transaction features.
                              Should have columns: V1-V28, Amount, Time
        
        Returns:
            pd.DataFrame: Original DataFrame with added prediction columns:
                         - 'fraud_prediction': 0 (legitimate) or 1 (fraud)
                         - 'fraud_probability': Probability of fraud (0-1)
        """
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        X = df[feature_cols].values
        
        results = self.predict_batch(X)
        
        df_results = df.copy()
        df_results['fraud_prediction'] = [r['prediction'] for r in results]
        df_results['fraud_probability'] = [r['confidence'] for r in results]
        
        return df_results


def load_model():
    """
    Convenience function to load model.
    
    Returns:
        FraudDetectionPredictor: Initialized predictor
    """
    return FraudDetectionPredictor()


# Example usage
if __name__ == "__main__":
    print("Loading model...")
    predictor = FraudDetectionPredictor()
    print("✓ Model loaded successfully\n")
    
    # Example: Single prediction
    print("Example 1: Single Transaction Prediction")
    print("-" * 50)
    example_features = np.random.randn(30)
    example_features[28] = 100  # Amount
    example_features[29] = 500  # Time
    
    result = predictor.predict(example_features)
    print(f"Transaction features shape: {example_features.shape}")
    print(f"Prediction: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    
    # Example: Batch prediction
    print("Example 2: Batch Prediction (5 transactions)")
    print("-" * 50)
    batch_features = np.random.randn(5, 30)
    batch_features[:, 28] = np.random.uniform(10, 1000, 5)  # Amount
    batch_features[:, 29] = np.random.uniform(0, 86400, 5)  # Time
    
    batch_results = predictor.predict_batch(batch_features)
    for i, result in enumerate(batch_results):
        print(f"Transaction {i+1}: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'} "
              f"(Confidence: {result['confidence']:.2%})")
