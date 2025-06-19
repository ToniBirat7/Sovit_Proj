import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    mean_squared_error, r2_score, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

def load_config(config_path: str = "/opt/airflow/config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evaluate_model(model: Any,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   config_path: str = "/opt/airflow/config/config.yaml") -> Dict[str, Any]:
    """Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        config_path: Path to the configuration file
        
    Returns:
        Dict containing evaluation metrics
    """
    print("Starting model evaluation process...")
    
    # Load configuration
    config = load_config(config_path)
    model_config = config['model']
    eval_metrics = model_config['evaluation_metrics']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Determine if this is a classification or regression task
    is_classification = len(y_test.unique()) < 10
    
    # Calculate metrics
    metrics = {}
    
    if is_classification:
        print("Evaluating classification model...")
        # For ROC AUC, we need probability predictions
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        if 'accuracy' in eval_metrics:
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        
        if 'precision' in eval_metrics:
            metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
        
        if 'recall' in eval_metrics:
            metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
        
        if 'f1' in eval_metrics:
            metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
        
        if 'roc_auc' in eval_metrics and y_pred_proba is not None:
            # Check if the problem is binary
            if len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
            else:
                print("ROC AUC is only applicable for binary classification")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate plots
        os.makedirs("/opt/airflow/data/evaluation", exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("/opt/airflow/data/evaluation/confusion_matrix.png")
        plt.close()
        
        # 2. ROC Curve (for binary classification)
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics.get("roc_auc", 0):.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig("/opt/airflow/data/evaluation/roc_curve.png")
            plt.close()
        
        # 3. Precision-Recall Curve (for binary classification)
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.savefig("/opt/airflow/data/evaluation/precision_recall_curve.png")
            plt.close()
    else:
        print("Evaluating regression model...")
        # Regression metrics
        metrics['mean_squared_error'] = float(mean_squared_error(y_test, y_pred))
        metrics['root_mean_squared_error'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['mean_absolute_error'] = float(mean_absolute_error(y_test, y_pred))
        metrics['r2_score'] = float(r2_score(y_test, y_pred))
        
        # Generate plots
        os.makedirs("/opt/airflow/data/evaluation", exist_ok=True)
        
        # 1. Actual vs Predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.savefig("/opt/airflow/data/evaluation/actual_vs_predicted.png")
        plt.close()
        
        # 2. Residuals Plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.savefig("/opt/airflow/data/evaluation/residuals_plot.png")
        plt.close()
        
        # 3. Residuals Distribution
        plt.figure(figsize=(10, 8))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.savefig("/opt/airflow/data/evaluation/residuals_distribution.png")
        plt.close()
    
    # 4. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importances = {
            feature: float(importance)
            for feature, importance in zip(X_test.columns, model.feature_importances_)
        }
        
        # Sort by importance
        feature_importances = {
            k: v for k, v in sorted(
                feature_importances.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        }
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        features = list(feature_importances.keys())[:20]
        importances = list(feature_importances.values())[:20]
        
        sns.barplot(x=importances, y=features)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig("/opt/airflow/data/evaluation/feature_importance.png")
        plt.close()
        
        # Add feature importances to metrics
        metrics['feature_importances'] = feature_importances
    
    # Save evaluation results
    results_path = "/opt/airflow/data/evaluation/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print metrics
    print("Model evaluation completed:")
    for metric, value in metrics.items():
        if metric != 'feature_importances':
            print(f"- {metric}: {value:.4f}")
    
    return metrics
