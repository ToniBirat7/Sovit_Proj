#!/bin/bash

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and essential build tools
echo "Upgrading pip and installing essential build tools..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize project directories
echo "Initializing project structure..."
mkdir -p airflow/dags airflow/plugins/operators
mkdir -p src/data src/features src/models src/utils
mkdir -p config

echo "Setup completed successfully!"
