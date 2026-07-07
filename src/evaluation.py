"""
Model evaluation module for Credit Card Fraud Detection.

This module provides:
- Comprehensive evaluation metrics
- Visualization functions
- Results summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)

from src.config import (
    COLOR_LEGIT, COLOR_FRAUD, COLOR_TITLE, FIGURE_DPI,
    OUTPUTS_DIR, EDA_PLOT_NAME, EVALUATION_PLOT_NAME
)


def evaluate_model(y_test, y_pred, y_proba, model_name="Model"):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
        y_test (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_proba (np.ndarray): Predicted probabilities for class 1
        model_name (str): Model name for logging
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Legitimate", "Fraud"],
            output_dict=True
        )
    }
    return metrics


def evaluate_all_models(models, X_test, y_test, y_probas, y_preds):
    """
    Evaluate all models and create comparison.
    
    Parameters:
        models (dict): Dictionary of model names
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        y_probas (dict): Dictionary of predicted probabilities for each model
        y_preds (dict): Dictionary of predicted labels for each model
    
    Returns:
        dict: Dictionary of {model_name: evaluation_metrics}
    """
    results = {}
    for model_name in models.keys():
        results[model_name] = evaluate_model(
            y_test, y_preds[model_name], y_probas[model_name],
            model_name=model_name
        )
    return results


def print_results_summary(results):
    """
    Print summary of model results in table format.
    
    Parameters:
        results (dict): Dictionary of {model_name: evaluation_metrics}
    """
    print("\n" + "─" * 70)
    print(f"{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>10}")
    print("─" * 70)
    
    best_f1 = 0
    best_model = None
    
    for name, metrics in results.items():
        f1 = metrics["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_model = name
        
        marker = " ◀ BEST" if name == best_model else ""
        print(
            f"{name:<22} {metrics['precision']:>9.2%}  "
            f"{metrics['recall']:>7.2%} {metrics['f1']:>7.2%}  "
            f"{metrics['roc_auc']:>9.4f}{marker}"
        )
    
    print("─" * 70)
    
    return best_model


def print_detailed_results(model_name, metrics):
    """
    Print detailed evaluation metrics for a single model.
    
    Parameters:
        model_name (str): Name of the model
        metrics (dict): Evaluation metrics dictionary
    """
    print(f"\n  Best Model: {model_name}")
    print(f"  ✓ Accuracy   : {metrics['accuracy']:.2%}")
    print(f"  ✓ Precision  : {metrics['precision']:.2%}")
    print(f"  ✓ Recall     : {metrics['recall']:.2%}")
    print(f"  ✓ F1-Score   : {metrics['f1']:.2%}")
    print(f"  ✓ ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"\n  Classification Report ({model_name}):")
    
    # Convert classification report dict to readable format
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    print(report_df.to_string())


def plot_eda(df, feature_cols):
    """
    Create comprehensive EDA visualizations.
    
    Parameters:
        df (pd.DataFrame): Dataset with features and 'Class' column
        feature_cols (list): List of feature column names
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Credit Card Fraud Detection — EDA",
        fontsize=15, fontweight='bold', color=COLOR_TITLE, y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    
    # (a) Class Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["Class"].value_counts()
    colors = [COLOR_LEGIT, COLOR_FRAUD]
    bars = ax1.bar(["Legitimate", "Fraud"], counts.values, color=colors, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f"{v:,}", ha='center', fontsize=10, fontweight='bold')
    ax1.set_title("Class Distribution", fontweight='bold', color=COLOR_TITLE)
    ax1.set_ylabel("Count")
    ax1.set_yscale('log')
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # (b) Transaction Amount Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df[df.Class==0]["Amount"].clip(upper=500), bins=50,
            color=COLOR_LEGIT, alpha=0.7, label="Legitimate", density=True)
    ax2.hist(df[df.Class==1]["Amount"].clip(upper=500), bins=30,
            color=COLOR_FRAUD, alpha=0.8, label="Fraud", density=True)
    ax2.set_title("Transaction Amount (clipped at $500)", fontweight='bold', color=COLOR_TITLE)
    ax2.set_xlabel("Amount ($)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # (c) Amount Distribution Boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    bp_data = [df[df.Class==0]["Amount"].clip(upper=1000).values,
               df[df.Class==1]["Amount"].clip(upper=1000).values]
    bp = ax3.boxplot(bp_data, labels=["Legitimate", "Fraud"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax3.set_title("Amount Distribution by Class", fontweight='bold', color=COLOR_TITLE)
    ax3.set_ylabel("Amount ($)")
    ax3.yaxis.grid(True, alpha=0.3)
    ax3.set_axisbelow(True)
    
    # (d) Correlation Heatmap (Top features)
    ax4 = fig.add_subplot(gs[1, 0:2])
    top_feats = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "Amount", "Class"]
    corr = df[top_feats].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax4, cmap="RdBu_r", center=0,
               annot=True, fmt=".2f", annot_kws={"size": 7.5},
               linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax4.set_title("Feature Correlation (Top 10)", fontweight='bold', color=COLOR_TITLE)
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.tick_params(axis='y', rotation=0, labelsize=8)
    
    # (e) Transaction Time
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(df[df.Class==0]["Time"]/3600, bins=48,
            color=COLOR_LEGIT, alpha=0.6, label="Legitimate", density=True)
    ax5.hist(df[df.Class==1]["Time"]/3600, bins=24,
            color=COLOR_FRAUD, alpha=0.8, label="Fraud", density=True)
    ax5.set_title("Transaction Time (Hours)", fontweight='bold', color=COLOR_TITLE)
    ax5.set_xlabel("Hour")
    ax5.set_ylabel("Density")
    ax5.legend()
    ax5.yaxis.grid(True, alpha=0.3)
    ax5.set_axisbelow(True)
    
    plt.savefig(f"{OUTPUTS_DIR}/{EDA_PLOT_NAME}", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✓ EDA plots saved to {OUTPUTS_DIR}/{EDA_PLOT_NAME}")


def plot_evaluation(results, y_test, X_test_predictions, best_model_name):
    """
    Create comprehensive model evaluation visualizations.
    
    Parameters:
        results (dict): Dictionary of {model_name: evaluation_metrics}
        y_test (np.ndarray): Test labels
        X_test_predictions (dict): Dictionary of {model_name: (y_pred, y_proba)}
        best_model_name (str): Name of the best performing model
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Credit Card Fraud Detection — Model Evaluation",
        fontsize=15, fontweight='bold', color=COLOR_TITLE, y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)
    
    COLORS = {
        "Logistic Regression": "#7CB9E8",
        "Random Forest": COLOR_LEGIT,
        "Gradient Boosting": COLOR_FRAUD
    }
    
    best_metrics = results[best_model_name]
    best_y_pred, best_y_proba = X_test_predictions[best_model_name]
    
    # (a) Confusion Matrix — Best Model
    ax1 = fig.add_subplot(gs[0, 0])
    cm = best_metrics["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'],
               linewidths=1, cbar=False, annot_kws={"size": 13, "fontweight": "bold"})
    ax1.set_title(f"Confusion Matrix\n({best_model_name})", fontweight='bold', color=COLOR_TITLE)
    ax1.set_ylabel("Actual")
    ax1.set_xlabel("Predicted")
    
    # (b) ROC Curves — All models
    ax2 = fig.add_subplot(gs[0, 1])
    for name, metrics in results.items():
        y_proba = X_test_predictions[name][1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax2.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})",
                color=COLORS[name], linewidth=2)
    ax2.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax2.set_title("ROC Curves — All Models", fontweight='bold', color=COLOR_TITLE)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    
    # (c) Precision-Recall Curve
    ax3 = fig.add_subplot(gs[0, 2])
    for name, metrics in results.items():
        y_proba = X_test_predictions[name][1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ax3.plot(rec, prec, label=f"{name} (AP={metrics['avg_precision']:.3f})",
                color=COLORS[name], linewidth=2)
    ax3.set_title("Precision-Recall Curves", fontweight='bold', color=COLOR_TITLE)
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.legend(fontsize=8.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.02])
    
    # (d) Metrics Comparison Bar Chart
    ax4 = fig.add_subplot(gs[1, 0:2])
    metric_names = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metric_names))
    width = 0.25
    
    for i, (name, metrics) in enumerate(results.items()):
        vals = [metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"]]
        bars = ax4.bar(x + i*width, vals, width, label=name, color=COLORS[name],
                      edgecolor='white', linewidth=0.8, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(metric_names, fontsize=10)
    ax4.set_ylim(0, 1.12)
    ax4.set_title("Model Performance Comparison", fontweight='bold', color=COLOR_TITLE)
    ax4.legend(fontsize=9)
    ax4.yaxis.grid(True, alpha=0.3)
    ax4.set_axisbelow(True)
    
    # (e) Feature Importance — Best Model (if available)
    ax5 = fig.add_subplot(gs[1, 2])
    if hasattr(results[best_model_name].get('model'), 'feature_importances_'):
        model = results[best_model_name].get('model')
        importances = model.feature_importances_
        feature_names = results[best_model_name].get('feature_names', 
                                                     [f"Feature_{i}" for i in range(len(importances))])
        top_idx = np.argsort(importances)[-10:][::-1]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = importances[top_idx]
        bars = ax5.barh(range(10), top_vals[::-1], color=COLOR_LEGIT, alpha=0.85, edgecolor='white')
        ax5.set_yticks(range(10))
        ax5.set_yticklabels(top_names[::-1], fontsize=9)
        ax5.set_title(f"Top 10 Feature Importances\n({best_model_name})",
                     fontweight='bold', color=COLOR_TITLE)
        ax5.set_xlabel("Importance")
        ax5.xaxis.grid(True, alpha=0.3)
        ax5.set_axisbelow(True)
    else:
        ax5.text(0.5, 0.5, f"{best_model_name}\ndoes not support feature importance",
                ha='center', va='center', fontsize=10, color='gray')
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    plt.savefig(f"{OUTPUTS_DIR}/{EVALUATION_PLOT_NAME}", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Evaluation plots saved to {OUTPUTS_DIR}/{EVALUATION_PLOT_NAME}")
