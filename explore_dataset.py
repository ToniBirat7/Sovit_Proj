#!/usr/bin/env python3
"""
Dataset Explorer for Lung Cancer Mortality Data
This script reads and analyzes the lung cancer mortality dataset, providing insights
about its structure, features, and basic statistics. It also updates the config.yaml
file with the appropriate feature names.
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # Define paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir
    dataset_path = project_root / "Dataset" / "lung_cancer_mortality_data_test_v2.csv"
    config_path = project_root / "config" / "config.yaml"
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)
    
    # Read the dataset
    print(f"Reading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)
    
    # Analyze dataset structure
    print("\n=== Dataset Structure ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Display column names and types
    print("\n=== Columns ===")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\n=== Missing Values ===")
        for col, count in missing_values[missing_values > 0].items():
            print(f"{col}: {count} ({count/len(df):.2%})")
    else:
        print("\nNo missing values found.")
    
    # Basic statistics for numeric columns
    print("\n=== Numeric Columns Statistics ===")
    print(df[numeric_cols].describe().transpose())
    
    # Find the target column (last column or user input)
    if len(sys.argv) > 1 and sys.argv[1] in df.columns:
        target_column = sys.argv[1]
    else:
        # Assume the last column is the target, but ask for confirmation
        suggested_target = df.columns[-1]
        response = input(f"\nIs '{suggested_target}' the target column? (y/n): ")
        if response.lower() in ['y', 'yes']:
            target_column = suggested_target
        else:
            print("\nAvailable columns:")
            for i, col in enumerate(df.columns):
                print(f"{i}: {col}")
            target_idx = int(input("\nEnter the index of the target column: "))
            target_column = df.columns[target_idx]
    
    print(f"\nUsing '{target_column}' as the target column")
    
    # Update the config file
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update target and feature lists
        config['data_processing']['target'] = target_column
        
        # Remove target from feature lists if present
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        config['data_processing']['features']['numerical'] = numeric_cols
        config['data_processing']['features']['categorical'] = categorical_cols
        
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"\nConfig file updated successfully: {config_path}")
    except Exception as e:
        print(f"Error updating config file: {e}")
    
    # Generate a simple report about the dataset
    output_dir = project_root / "Dataset" / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Save dataset summary
    with open(output_dir / "dataset_summary.txt", 'w') as f:
        f.write(f"Dataset: {dataset_path.name}\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("=== Columns ===\n")
        for col in df.columns:
            f.write(f"{col}: {df[col].dtype}\n")
        
        f.write(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}\n")
        f.write(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}\n")
        f.write(f"Target column: {target_column}\n\n")
        
        if missing_values.sum() > 0:
            f.write("=== Missing Values ===\n")
            for col, count in missing_values[missing_values > 0].items():
                f.write(f"{col}: {count} ({count/len(df):.2%})\n")
        else:
            f.write("No missing values found.\n")
    
    print(f"\nDataset analysis saved to: {output_dir / 'dataset_summary.txt'}")
    
    print("\nDataset exploration completed successfully!")

if __name__ == "__main__":
    main()
