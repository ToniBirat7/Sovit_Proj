import pandas as pd
import numpy as np
import yaml
import os
from typing import Tuple, Dict, Any, List, Optional

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

def read_csv_data(config_path: str = "/opt/airflow/config/config.yaml") -> pd.DataFrame:
    """Read the raw CSV data.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        DataFrame containing the raw data
    """
    config = load_config(config_path)
    file_path = config['paths']['raw_data']
    
    print(f"Reading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("Starting data cleaning process...")
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # 1. Handle missing values
    missing_before = df_cleaned.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # For numeric columns, fill missing values with median
    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # For categorical columns, fill missing values with mode
    cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown")
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    
    # 2. Remove duplicates
    duplicates_before = df_cleaned.duplicated().sum()
    print(f"Duplicates before cleaning: {duplicates_before}")
    
    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    
    duplicates_after = df_cleaned.duplicated().sum()
    print(f"Duplicates after cleaning: {duplicates_after}")
    
    # 3. Check for outliers in numeric columns (using IQR method)
    for col in numeric_cols:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Count outliers
        outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Column '{col}' has {outliers} outliers")
            
            # Cap outliers
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
    
    print(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
    return df_cleaned

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary of the dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dict containing dataset summary information
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "summary_stats": df.describe().to_dict()
    }
    
    return summary
