import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import sys
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

def split_data(df: pd.DataFrame, 
               target_column: str,
               config_path: str = "/opt/airflow/config/config.yaml") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data into training and testing sets.
    
    Args:
        df: Processed DataFrame
        target_column: Name of the target column
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("Splitting data into training and testing sets...")
    
    # Load configuration
    config = load_config(config_path)
    test_size = config['data_processing']['test_size']
    random_state = config['data_processing']['random_state']
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")
    
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 10 else None
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_test, y_test

def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                config_path: str = "/opt/airflow/config/config.yaml") -> Tuple[Any, Dict[str, Any]]:
    """Train a model on the training data.
    
    Args:
        X_train: Training features
        y_train: Training target
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (trained model, training metadata)
    """
    print("Starting model training process...")
    
    # Load configuration
    config = load_config(config_path)
    model_config = config['model']
    algorithm = model_config['algorithm']
    hyperparameters = model_config['hyperparameters']
    
    # Initialize model based on algorithm
    if algorithm == "RandomForestClassifier":
        model = RandomForestClassifier(**hyperparameters)
    elif algorithm == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(**hyperparameters)
    elif algorithm == "LogisticRegression":
        model = LogisticRegression(**hyperparameters)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train model
    print(f"Training {algorithm} model...")
    model.fit(X_train, y_train)
    
    # Get feature importances if available
    feature_importances = {}
    if hasattr(model, 'feature_importances_'):
        feature_importances = {
            feature: importance
            for feature, importance in zip(X_train.columns, model.feature_importances_)
        }
        
        # Sort by importance
        feature_importances = {
            k: v for k, v in sorted(
                feature_importances.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        }
    
    # Create training metadata
    training_meta = {
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "feature_importances": feature_importances,
        "training_samples": X_train.shape[0],
        "features": list(X_train.columns),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save model
    os.makedirs("/opt/airflow/models", exist_ok=True)
    model_path = f"/opt/airflow/models/{algorithm.lower()}_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training metadata
    meta_path = f"/opt/airflow/models/{algorithm.lower()}_metadata.json"
    with open(meta_path, 'w') as f:
        # Convert numpy types to Python native types for JSON serialization
        training_meta_json = {
            k: v if not isinstance(v, dict) else {
                inner_k: float(inner_v) if isinstance(inner_v, np.floating) else inner_v
                for inner_k, inner_v in v.items()
            }
            for k, v in training_meta.items()
        }
        json.dump(training_meta_json, f, indent=2)
    
    print("Model training completed")
    return model, training_meta

def grid_search_cv(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   config_path: str = "/opt/airflow/config/config.yaml") -> Tuple[Any, Dict[str, Any]]:
    """Perform grid search cross-validation to find the best hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (best model, hyperparameter search results)
    """
    print("Starting hyperparameter tuning with grid search...")
    
    # Load configuration
    config = load_config(config_path)
    model_config = config['model']
    algorithm = model_config['algorithm']
    
    # Define parameter grid based on algorithm
    if algorithm == "RandomForestClassifier":
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        }
    elif algorithm == "GradientBoostingClassifier":
        model = GradientBoostingClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'random_state': [42]
        }
    elif algorithm == "LogisticRegression":
        model = LogisticRegression()
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2'],
            'random_state': [42]
        }
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Perform grid search
    print("Performing grid search cross-validation...")
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    
    # Create grid search results
    search_results = {
        "algorithm": algorithm,
        "best_parameters": best_params,
        "best_score": grid_search.best_score_,
        "all_scores": {
            str(params): score
            for params, score in zip(
                cv_results['params'],
                cv_results['mean_test_score']
            )
        }
    }
    
    # Save best model
    os.makedirs("/opt/airflow/models", exist_ok=True)
    model_path = f"/opt/airflow/models/{algorithm.lower()}_best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Best model saved to: {model_path}")
    
    # Save grid search results
    results_path = f"/opt/airflow/models/{algorithm.lower()}_grid_search_results.json"
    with open(results_path, 'w') as f:
        json.dump(search_results, f, indent=2)
    
    print(f"Hyperparameter tuning completed. Best parameters: {best_params}")
    return best_model, search_results
