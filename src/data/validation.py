import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Tuple, Optional
import json
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

def validate_data(df: pd.DataFrame, 
                  transform_meta: Dict[str, Any],
                  config_path: str = "/opt/airflow/config/config.yaml") -> Tuple[bool, Dict[str, Any]]:
    """Validate the transformed data.
    
    Args:
        df: Transformed DataFrame
        transform_meta: Metadata from the transformation step
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (validation passed, validation results)
    """
    print("Starting data validation process...")
    
    # Load configuration
    config = load_config(config_path)
    
    validation_results = {
        "passed": True,
        "expectations": [],
        "summary": {},
        "warnings": []
    }
    
    # 1. Check for missing values
    print("Checking for missing values...")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        validation_results["passed"] = False
        validation_results["warnings"].append("Missing values found in the dataset")
        validation_results["expectations"].append({
            "type": "missing_values_check",
            "success": False,
            "details": missing_values[missing_values > 0].to_dict()
        })
    else:
        validation_results["expectations"].append({
            "type": "missing_values_check",
            "success": True
        })
    
    # 2. Check for data types
    print("Checking data types...")
    numeric_features = transform_meta.get("numeric_features", [])
    for col in numeric_features:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_results["passed"] = False
                validation_results["warnings"].append(f"Column {col} expected to be numeric but is {df[col].dtype}")
                validation_results["expectations"].append({
                    "type": "data_type_check",
                    "column": col,
                    "success": False,
                    "expected_type": "numeric",
                    "actual_type": str(df[col].dtype)
                })
            else:
                validation_results["expectations"].append({
                    "type": "data_type_check",
                    "column": col,
                    "success": True,
                    "expected_type": "numeric",
                    "actual_type": str(df[col].dtype)
                })
    
    categorical_features = transform_meta.get("categorical_features", [])
    for col in categorical_features:
        if col in df.columns:
            if not pd.api.types.is_categorical_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                validation_results["passed"] = False
                validation_results["warnings"].append(f"Column {col} expected to be categorical but is {df[col].dtype}")
                validation_results["expectations"].append({
                    "type": "data_type_check",
                    "column": col,
                    "success": False,
                    "expected_type": "categorical",
                    "actual_type": str(df[col].dtype)
                })
            else:
                validation_results["expectations"].append({
                    "type": "data_type_check",
                    "column": col, 
                    "success": True,
                    "expected_type": "categorical",
                    "actual_type": str(df[col].dtype)
                })
    
    # 3. Check required columns
    print("Checking required columns...")
    required_columns = config.get("features", []) + [config.get("target", "")]
    required_columns = [col for col in required_columns if col]  # Remove empty strings
    
    for col in required_columns:
        if col not in df.columns:
            validation_results["passed"] = False
            validation_results["warnings"].append(f"Required column {col} is missing")
            validation_results["expectations"].append({
                "type": "column_exists_check",
                "column": col,
                "success": False
            })
        else:
            validation_results["expectations"].append({
                "type": "column_exists_check",
                "column": col,
                "success": True
            })
    
    # 4. Check for duplicate rows
    print("Checking for duplicate rows...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation_results["passed"] = False
        validation_results["warnings"].append(f"Found {duplicates} duplicate rows in the dataset")
    
    # 6. Check dataset size
    print("Checking dataset size...")
    min_rows = 10  # Minimum number of rows required
    if len(df) < min_rows:
        validation_results["passed"] = False
        validation_results["warnings"].append(f"Dataset too small: {len(df)} rows (minimum {min_rows})")
    
    # 7. Check feature correlations (to detect multicollinearity)
    print("Checking for multicollinearity...")
    corr_matrix = df.corr()
    high_corr_pairs = []
    
    # Find highly correlated features
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:  # Threshold for high correlation
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        validation_results["warnings"].append(f"Found {len(high_corr_pairs)} pairs of highly correlated features")
        validation_results["summary"]["high_corr_pairs"] = [
            {"feature1": str(pair[0]), "feature2": str(pair[1]), "correlation": float(pair[2])} 
            for pair in high_corr_pairs
        ]
    
    # 5. Check for value ranges for numeric features
    print("Checking value ranges...")
    for col in numeric_features:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            validation_results["summary"][col] = {
                "min": float(min_val),
                "max": float(max_val),
                "mean": float(mean_val),
                "std": float(std_val)
            }
    
    # 6. Check for distribution of categorical features
    print("Checking categorical distributions...")
    for col in categorical_features:
        if col in df.columns:
            value_counts = df[col].value_counts().to_dict()
            validation_results["summary"][col] = {
                "unique_values": len(value_counts),
                "top_categories": {str(k): int(v) for k, v in list(value_counts.items())[:5]}
            }
    
    # Print validation summary
    print(f"Data validation completed. Passed: {validation_results['passed']}")
    print(f"Warnings: {len(validation_results['warnings'])}")
    for warning in validation_results["warnings"]:
        print(f"- {warning}")
    
    # Save validation results to file
    os.makedirs("/opt/airflow/data/validation", exist_ok=True)
    validation_file = "/opt/airflow/data/validation/validation_results.json"
    
    # We need to make validation results JSON serializable
    simplified_results = {
        "passed": validation_results["passed"],
        "warnings": validation_results["warnings"],
        "summary": validation_results["summary"]
    }
    
    with open(validation_file, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    return validation_results["passed"], validation_results
