#!/bin/bash

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for Python 3.8+
if ! command_exists python3; then
  echo "Error: Python 3 is required but not installed."
  echo "Please install Python 3.8 or newer and try again."
  exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python_major_version=$(echo $python_version | cut -d. -f1)
python_minor_version=$(echo $python_version | cut -d. -f2)

if [ "$python_major_version" -lt 3 ] || ([ "$python_major_version" -eq 3 ] && [ "$python_minor_version" -lt 8 ]); then
  echo "Error: Python 3.8+ is required, but you have Python $python_version."
  echo "Please install Python 3.8 or newer and try again."
  exit 1
fi

echo "Using Python $python_version"

# Check for Docker and Docker Compose
if ! command_exists docker; then
  echo "Error: Docker is required but not installed."
  echo "Please install Docker and try again."
  exit 1
fi

if ! command_exists docker-compose; then
  echo "Error: Docker Compose is required but not installed."
  echo "Please install Docker Compose and try again."
  exit 1
fi

# Create and set up virtual environment
if [ ! -d "venv" ]; then
  echo "Setting up Python virtual environment..."
  # Create virtual environment
  python3 -m venv venv
  
  # Activate virtual environment
  source venv/bin/activate
  
  # Upgrade pip and install essential build tools
  echo "Upgrading pip and installing essential build tools..."
  pip install --upgrade pip setuptools wheel
  
  # Install dependencies
  echo "Installing dependencies..."
  pip install -r requirements.txt
else
  echo "Virtual environment already exists. Activating it..."
  source venv/bin/activate
fi

# Initialize project directories
echo "Ensuring project structure is set up..."
mkdir -p airflow/dags airflow/plugins/operators
mkdir -p src/data src/features src/models src/utils
mkdir -p config
mkdir -p logs mlflow

# Start Docker containers
echo "Starting Docker containers..."
docker-compose up -d

echo "Waiting for services to be fully initialized..."
sleep 10

echo "==============================================="
echo "🚀 MLOps Pipeline Setup Complete! 🚀"
echo "==============================================="
echo "Airflow UI:    http://localhost:8080 (Username: airflow, Password: airflow)"
echo "MLflow UI:     http://localhost:5000"
echo "Redis:         localhost:6379"
echo ""
echo "To stop the services, run: docker-compose down"
echo "To restart the services, run: docker-compose up -d"
echo ""
echo "Happy ML Engineering! 🎉"
