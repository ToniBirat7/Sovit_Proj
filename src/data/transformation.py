import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

def transform_data(df: pd.DataFrame, config_path: str = "/opt/airflow/config/config.yaml") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Transform the cleaned data.
    
    Args:
        df: Cleaned DataFrame
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (transformed DataFrame, transformation metadata)
    """
    print("Starting data transformation process...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Make a copy to avoid modifying the original
    df_transformed = df.copy()
    
    # Feature engineering and transformation metadata
    transform_meta = {
        "numeric_features": [],
        "categorical_features": [],
        "derived_features": [],
        "dropped_features": []
    }
    
    # Identify numeric and categorical features
    numeric_features = list(df_transformed.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = list(df_transformed.select_dtypes(include=['object', 'category']).columns)
    
    transform_meta["numeric_features"] = numeric_features
    transform_meta["categorical_features"] = categorical_features
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # 1. Handle categorical features - convert to one-hot encoding
    for col in categorical_features:
        # Check the number of unique values
        unique_count = df_transformed[col].nunique()
        
        if unique_count <= 15:  # Only one-hot encode if relatively few categories
            print(f"One-hot encoding column '{col}' with {unique_count} unique values")
            # Get dummies (one-hot encoding)
            dummies = pd.get_dummies(df_transformed[col], prefix=col, drop_first=True)
            # Join the dummy columns to the original dataframe
            df_transformed = pd.concat([df_transformed, dummies], axis=1)
            # Drop the original categorical column
            df_transformed = df_transformed.drop(col, axis=1)
            transform_meta["dropped_features"].append(col)
        else:
            print(f"Too many unique values ({unique_count}) for one-hot encoding column '{col}'. Considering label encoding instead.")
            # For simplicity, here we just drop high-cardinality categorical features
            # In a real project, you might want to use label encoding or target encoding
            df_transformed = df_transformed.drop(col, axis=1)
            transform_meta["dropped_features"].append(col)
    
    # 2. Create interaction features for numeric columns (if applicable)
    if len(numeric_features) >= 2:
        # Example: create a ratio feature from two numeric columns
        # Here I'm just using the first two numeric features as an example
        feature1 = numeric_features[0]
        feature2 = numeric_features[1]
        
        # Avoid division by zero
        df_transformed[f'{feature1}_to_{feature2}_ratio'] = df_transformed[feature1] / (df_transformed[feature2] + 1e-8)
        transform_meta["derived_features"].append(f'{feature1}_to_{feature2}_ratio')
        
        print(f"Created derived feature: {feature1}_to_{feature2}_ratio")
    
    # 3. Normalize numeric features
    scaler = StandardScaler()
    for col in numeric_features:
        df_transformed[col] = scaler.fit_transform(df_transformed[[col]])
        print(f"Normalized numeric feature: {col}")
    
    # Save scaler for future use
    os.makedirs("/opt/airflow/models", exist_ok=True)
    joblib.dump(scaler, "/opt/airflow/models/scaler.joblib")
    
    # 4. Log transformation for skewed numeric features
    for col in numeric_features:
        # Check for skewness
        skewness = df_transformed[col].skew()
        if abs(skewness) > 1:
            # Apply log transformation (adding 1 to handle zeros)
            new_col_name = f"{col}_log"
            df_transformed[new_col_name] = np.log1p(df_transformed[col].clip(lower=0))
            transform_meta["derived_features"].append(new_col_name)
            print(f"Applied log transformation to skewed feature '{col}' (skewness: {skewness:.2f})")
    
    print(f"Data transformation completed. Final shape: {df_transformed.shape}")
    return df_transformed, transform_meta
