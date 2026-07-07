"""
Training script for Credit Card Fraud Detection model.

This script:
1. Generates or loads dataset
2. Performs exploratory data analysis
3. Preprocesses data (with NO data leakage)
4. Trains multiple models
5. Evaluates and selects the best model
6. Saves the best model and scaler for production use

Usage:
    python train.py
"""

import sys
import os
import joblib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_kaggle_data, print_dataset_info, print_class_balance
from src.preprocessing import preprocess_pipeline
from src.models import get_models, train_models, predict_with_threshold
from src.evaluation import (
    evaluate_all_models, print_results_summary, print_detailed_results,
    plot_eda, plot_evaluation
)
from src.config import (
    MODELS_DIR, BEST_MODEL_NAME, SCALER_NAME
)


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("   CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load Dataset
    # =========================================================================
    print("\n[1/6] Loading Kaggle Credit Card Fraud Detection Dataset...")
    try:
        df = load_kaggle_data()
    except FileNotFoundError as e:
        print(str(e))
        return
    
    print_dataset_info(df)
    
    # =========================================================================
    # Step 2: Exploratory Data Analysis
    # =========================================================================
    print("\n[2/6] Exploratory Data Analysis...")
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    plot_eda(df, feature_cols)
    
    # =========================================================================
    # Step 3: Data Preprocessing
    # =========================================================================
    print("\n[3/6] Data Preprocessing & Feature Engineering...")
    print("   Note: Data split BEFORE scaling to prevent data leakage")
    
    preprocessed = preprocess_pipeline(df)
    
    X_train_res = preprocessed["X_train_res"]
    X_test = preprocessed["X_test"]
    y_train_res = preprocessed["y_train_res"]
    y_test = preprocessed["y_test"]
    scaler = preprocessed["scaler"]
    feature_columns = preprocessed["feature_columns"]
    
    print(f"   ✓ Training set: {len(X_train_res):,} samples (resampled)")
    print(f"   ✓ Test set: {len(X_test):,} samples")
    print("   ✓ Class distribution after resampling:")
    print_class_balance(y_train_res)
    
    # =========================================================================
    # Step 4: Model Training
    # =========================================================================
    print("\n[4/6] Training Models...")
    
    models = get_models()
    trained_models = train_models(models, X_train_res, y_train_res)
    
    # =========================================================================
    # Step 5: Model Evaluation
    # =========================================================================
    print("\n[5/6] Evaluating Models...")
    
    # Make predictions for all models
    y_probas = {}
    y_preds = {}
    X_test_predictions = {}
    
    for model_name, model in trained_models.items():
        y_pred, y_proba, threshold = predict_with_threshold(
            model, X_test, X_test, y_test, threshold=None
        )
        y_probas[model_name] = y_proba
        y_preds[model_name] = y_pred
        X_test_predictions[model_name] = (y_pred, y_proba)
        
        if model_name == "Gradient Boosting":
            print(f"   → {model_name}: Optimal threshold = {threshold:.3f}")
    
    # Evaluate all models
    results = evaluate_all_models(
        trained_models, X_test, y_test, y_probas, y_preds
    )
    
    # Print results
    best_model_name = print_results_summary(results)
    print()
    best_metrics = results[best_model_name]
    print_detailed_results(best_model_name, best_metrics)
    
    # =========================================================================
    # Step 6: Save Best Model
    # =========================================================================
    print("\n[6/6] Saving Best Model...")
    
    best_model = trained_models[best_model_name]
    
    # Save model
    model_path = os.path.join(MODELS_DIR, BEST_MODEL_NAME)
    joblib.dump(best_model, model_path)
    print(f"   ✓ Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, SCALER_NAME)
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Scaler saved to {scaler_path}")
    
    # Save feature columns
    features_path = os.path.join(MODELS_DIR, "feature_columns.joblib")
    joblib.dump(feature_columns, features_path)
    print(f"   ✓ Feature columns saved to {features_path}")
    
    # Create evaluation plots
    print("\n   Creating evaluation plots...")
    for model_name, model in trained_models.items():
        results[model_name]['model'] = model
        results[model_name]['feature_names'] = feature_columns
    
    plot_evaluation(results, y_test, X_test_predictions, best_model_name)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("   TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Best Model: {best_model_name}")
    print(f"  ✓ Accuracy  : {best_metrics['accuracy']:.2%}")
    print(f"  ✓ Precision : {best_metrics['precision']:.2%}")
    print(f"  ✓ Recall    : {best_metrics['recall']:.2%}")
    print(f"  ✓ F1-Score  : {best_metrics['f1']:.2%}")
    print(f"  ✓ ROC-AUC   : {best_metrics['roc_auc']:.4f}")
    print(f"\n  Saved Artifacts:")
    print(f"  - Model:  {model_path}")
    print(f"  - Scaler: {scaler_path}")
    print(f"  - Features: {features_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
