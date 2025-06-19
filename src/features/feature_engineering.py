import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

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

def engineer_features(df: pd.DataFrame, config_path: str = "/opt/airflow/config/config.yaml") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform feature engineering on the transformed data.
    
    Args:
        df: Transformed DataFrame
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (DataFrame with engineered features, feature engineering metadata)
    """
    print("Starting feature engineering process...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Make a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # Feature engineering metadata
    fe_meta = {
        "original_shape": df.shape,
        "created_features": [],
        "selected_features": [],
        "feature_engineering_methods": []
    }
    
    # 1. Generate polynomial features for numeric columns
    numeric_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) >= 2:
        # Select a subset of numeric columns to avoid explosion of features
        if len(numeric_cols) > 5:
            selected_numeric = list(numeric_cols)[:5]  # Take the first 5 numeric columns
        else:
            selected_numeric = list(numeric_cols)
        
        print(f"Generating polynomial features for {len(selected_numeric)} numeric columns")
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df_engineered[selected_numeric])
        
        # Get feature names
        poly_feature_names = []
        for i, feat in enumerate(poly.get_feature_names_out(selected_numeric)):
            if i >= len(selected_numeric):  # Skip the original features
                poly_feature_names.append(f"poly_{feat}")
        
        # Add polynomial features to DataFrame
        poly_df = pd.DataFrame(
            poly_features[:, len(selected_numeric):],  # Skip the original features
            columns=poly_feature_names,
            index=df_engineered.index
        )
        
        df_engineered = pd.concat([df_engineered, poly_df], axis=1)
        
        fe_meta["created_features"].extend(poly_feature_names)
        fe_meta["feature_engineering_methods"].append("polynomial_features")
        
        # Save polynomial features transformer
        os.makedirs("/opt/airflow/models", exist_ok=True)
        joblib.dump(poly, "/opt/airflow/models/poly_features.joblib")
    
    # 2. Feature selection - select top k features
    # We'll use SelectKBest with f_classif for classification tasks
    
    # Assume the last column is the target (this should be adapted based on your specific dataset)
    X = df_engineered.iloc[:, :-1]
    y = df_engineered.iloc[:, -1]
    
    # Select top k features (where k is half of the total features, or 50, whichever is smaller)
    k = min(X.shape[1] // 2, 50)
    
    print(f"Selecting top {k} features using SelectKBest")
    
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get the mask of selected features
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    fe_meta["selected_features"] = selected_features
    fe_meta["feature_engineering_methods"].append("select_k_best")
    
    # Create a new DataFrame with only selected features plus the target
    df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df_engineered.index)
    df_selected[df_engineered.columns[-1]] = df_engineered.iloc[:, -1]  # Add target column
    
    # Save feature selector
    joblib.dump(selector, "/opt/airflow/models/feature_selector.joblib")
    
    # 3. Add aggregate features
    # For example, sum, mean, min, max of selected numeric features
    numeric_selected = [col for col in selected_features if df_selected[col].dtype in ['int64', 'float64']]
    
    if len(numeric_selected) >= 2:
        print("Creating aggregate features")
        
        # Add sum of all numeric features
        df_selected['sum_numeric'] = df_selected[numeric_selected].sum(axis=1)
        fe_meta["created_features"].append('sum_numeric')
        
        # Add mean of all numeric features
        df_selected['mean_numeric'] = df_selected[numeric_selected].mean(axis=1)
        fe_meta["created_features"].append('mean_numeric')
        
        # Add min and max of all numeric features
        df_selected['min_numeric'] = df_selected[numeric_selected].min(axis=1)
        fe_meta["created_features"].append('min_numeric')
        
        df_selected['max_numeric'] = df_selected[numeric_selected].max(axis=1)
        fe_meta["created_features"].append('max_numeric')
        
        fe_meta["feature_engineering_methods"].append("aggregate_features")
    
    fe_meta["final_shape"] = df_selected.shape
    fe_meta["final_feature_count"] = df_selected.shape[1] - 1  # Subtract 1 for the target column
    
    print(f"Feature engineering completed. Final shape: {df_selected.shape}")
    return df_selected, fe_meta
