# 🚀 MLOps Pipeline Setup and Execution Guide

This guide provides step-by-step instructions to clone, setup, and run the complete MLOps pipeline for lung cancer mortality prediction using Airflow, MLflow, and Redis.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+, or similar)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **Internet**: Required for downloading Docker images and dependencies

### Required Software

- **Docker**: Version 20.10+
- **Docker Compose**: Version 1.29+
- **Git**: For cloning the repository
- **curl**: For testing APIs (usually pre-installed)

## 🛠️ Installation Steps

### Step 1: Install Docker and Docker Compose

#### For Ubuntu/Debian:

```bash
# Add current user to docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Verify installations
docker --version
docker-compose --version
```

### Step 2: Clone the Repository

```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-name>

# Verify you're in the correct directory
ls -la
# You should see: docker-compose.yml, airflow/, src/, Dataset/, etc.
```

### Step 3: Set Up the Environment

```bash
# Make setup scripts executable
chmod +x setup.sh start_project.sh

# Create required directories and set permissions
mkdir -p logs mlflow_artifacts
chmod 777 logs mlflow_artifacts

# Verify the dataset exists
ls -la Dataset/
# You should see: lung_cancer_mortality_data_test_v2.csv
```

## 🚀 Running the Pipeline

### Step 4: Start All Services

```bash
# Start all Docker containers
docker-compose up -d

# Wait for all services to be healthy (this may take 2-3 minutes)
echo "Waiting for services to start..."
sleep 60

# Check service status
docker-compose ps
```

**Expected Output**: All services should show "Up" or "Up (healthy)" status:

- `airflow-webserver` (Port 8080)
- `airflow-scheduler`
- `airflow-worker`
- `mlflow` (Port 5000)
- `redis` (Port 6379)
- `postgres` (Port 5432)

### Step 5: Install Required Python Packages in Airflow Containers

**⚠️ Critical Step**: The Airflow containers need additional Python packages installed:

```bash
# Install packages in airflow-scheduler
docker exec -it $(docker-compose ps -q airflow-scheduler) pip3 install mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib

# Install packages in airflow-webserver
docker exec -it $(docker-compose ps -q airflow-webserver) pip3 install mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib

# Install packages in airflow-worker
docker exec -it $(docker-compose ps -q airflow-worker) pip3 install mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib
```

### Step 6: Create Airflow Admin User

```bash
# Create admin user for Airflow UI
docker exec -it $(docker-compose ps -q airflow-webserver) airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Step 7: Copy Source Files to Airflow Containers

```bash
# Copy all source files to each Airflow container
for container in airflow-scheduler airflow-webserver airflow-worker; do
    echo "Copying files to $container..."
    docker cp src/ $(docker-compose ps -q $container):/opt/airflow/
    docker cp config/ $(docker-compose ps -q $container):/opt/airflow/
    docker cp Dataset/ $(docker-compose ps -q $container):/opt/airflow/
done
```

### Step 8: Set Up Directories and Permissions in Containers

```bash
# Create required directories in all Airflow containers
for container in airflow-scheduler airflow-webserver airflow-worker; do
    echo "Setting up directories in $container..."
    docker exec -it $(docker-compose ps -q $container) mkdir -p /opt/airflow/models /opt/airflow/mlflow_artifacts
    docker exec -it $(docker-compose ps -q $container) chmod 777 /opt/airflow/models /opt/airflow/mlflow_artifacts
done
```

### Step 9: Verify DAG Registration

```bash
# Check if the DAG is registered
docker exec -it $(docker-compose ps -q airflow-webserver) airflow dags list | grep lung_cancer

# If not visible, trigger DAG refresh
docker exec -it $(docker-compose ps -q airflow-scheduler) airflow dags list-import-errors
```

### Step 10: Access the Services

#### Airflow Web UI

- **URL**: http://localhost:8080
- **Username**: admin
- **Password**: admin

#### MLflow UI

- **URL**: http://localhost:5000
- **No authentication required**

#### Verify Services Are Running

```bash
# Test Airflow API
curl -I http://localhost:8080

# Test MLflow API
curl -I http://localhost:5000

# Test Redis
docker exec -it $(docker-compose ps -q redis) redis-cli ping
```

## 🎯 Running the ML Pipeline

### Step 11: Trigger the Pipeline

#### Option 1: Using Airflow UI (Recommended)

1. Open http://localhost:8080
2. Login with admin/admin
3. Find the DAG: `lung_cancer_ml_pipeline`
4. Toggle the DAG to "ON" (unpause it)
5. Click the "Play" button to trigger manually

#### Option 2: Using Command Line

```bash
# Unpause the DAG
docker exec -it $(docker-compose ps -q airflow-webserver) airflow dags unpause lung_cancer_ml_pipeline

# Trigger the DAG
docker exec -it $(docker-compose ps -q airflow-webserver) airflow dags trigger lung_cancer_ml_pipeline
```

### Step 12: Monitor Pipeline Execution

#### Monitor DAG Status

```bash
# Check DAG runs
docker exec -it $(docker-compose ps -q airflow-webserver) airflow dags list-runs -d lung_cancer_ml_pipeline

# Check task status for a specific run
docker exec -it $(docker-compose ps -q airflow-webserver) airflow tasks states-for-dag-run lung_cancer_ml_pipeline <RUN_ID>
```

#### Monitor Individual Tasks

The pipeline consists of 8 tasks that run sequentially:

1. **ingest_data_task** - Load raw data from CSV
2. **clean_data_task** - Clean and preprocess data
3. **transform_data_task** - Apply transformations
4. **validate_data_task** - Validate data quality
5. **feature_engineering_task** - Create engineered features
6. **train_model_task** - Train machine learning model
7. **evaluate_model_task** - Evaluate model performance
8. **register_model_task** - Register model in MLflow

#### Check Results

```bash
# Check if models were saved
docker exec -it $(docker-compose ps -q airflow-worker) ls -la /opt/airflow/models/

# View model metadata
docker exec -it $(docker-compose ps -q airflow-worker) cat /opt/airflow/models/randomforestregressor_metadata.json
```

## 🔧 Troubleshooting

### 🚀 Quick Fix Tool

If you encounter any issues after setup, use the interactive fix tool:

```bash
./fix_issues.sh
```

This tool can help with:

- Python package import issues
- Service restart problems
- DAG registration issues
- File permission problems
- Complete health checks

### Common Issues and Solutions

#### 0. Python Package Import Issues (Most Common)

**Symptoms**: `verify_setup.sh` shows "Missing packages" for scheduler container

**Quick Fix**:

```bash
./fix_issues.sh
# Select option 1: Fix Python package import issues
```

**Manual Fix**:

```bash
# Reinstall packages in scheduler container
docker exec $(docker-compose ps -q airflow-scheduler) pip3 install mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib --force-reinstall
```

#### 1. Docker Permission Denied

```bash
# If you get permission denied errors
sudo chmod 666 /var/run/docker.sock
# OR add user to docker group and restart terminal
```

#### 2. Port Already in Use

```bash
# Check what's using the ports
sudo netstat -tulpn | grep :8080
sudo netstat -tulpn | grep :5000

# Stop conflicting services or change ports in docker-compose.yml
```

#### 3. Services Not Starting

```bash
# Check logs for specific service
docker-compose logs airflow-webserver
docker-compose logs mlflow
docker-compose logs redis

# Restart services
docker-compose restart
```

#### 4. DAG Not Visible

```bash
# Check DAG import errors
docker exec -it $(docker-compose ps -q airflow-webserver) airflow dags list-import-errors

# Manually refresh DAGs
docker exec -it $(docker-compose ps -q airflow-scheduler) airflow dags reserialize
```

#### 5. Python Package Import Errors

```bash
# Reinstall packages in all containers
for container in airflow-scheduler airflow-webserver airflow-worker; do
    docker exec -it $(docker-compose ps -q $container) pip3 install mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib
done
```

#### 6. MLflow Permission Errors

The pipeline includes robust error handling for MLflow issues. Models will be saved locally even if MLflow fails:

```bash
# Check local model storage
docker exec -it $(docker-compose ps -q airflow-worker) ls -la /opt/airflow/models/
```

#### 7. Memory Issues

```bash
# If containers are killed due to memory issues
# Increase Docker memory limit in Docker Desktop settings
# Or reduce the number of workers in docker-compose.yml
```

### Log Locations

```bash
# Airflow logs
docker-compose logs airflow-webserver
docker-compose logs airflow-scheduler
docker-compose logs airflow-worker

# MLflow logs
docker-compose logs mlflow

# Application logs
docker exec -it $(docker-compose ps -q airflow-worker) ls /opt/airflow/logs/
```

## 🛑 Stopping the Pipeline

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: This deletes all data)
docker-compose down -v

# Remove all containers and images (complete cleanup)
docker-compose down --rmi all -v
```

## 📊 Expected Results

### Successful Pipeline Run Should Show:

- ✅ All 8 tasks completed successfully
- ✅ Models saved in `/opt/airflow/models/`
- ✅ MLflow experiments visible at http://localhost:5000
- ✅ Feature importance analysis available
- ✅ Model evaluation metrics generated

### Sample Model Output:

```json
{
  "algorithm": "RandomForestRegressor",
  "training_samples": 800,
  "features": 20,
  "top_feature": "id_to_age_ratio (99.5% importance)",
  "model_type": "regression"
}
```

## 🆘 Getting Help

If you encounter issues:

1. **Check this troubleshooting section first**
2. **Review the logs** using the commands provided
3. **Verify all prerequisites** are installed correctly
4. **Ensure sufficient system resources** (RAM, disk space)
5. **Check Docker and Docker Compose versions**

## 📝 Notes

- **First run** may take 10-15 minutes due to Docker image downloads
- **Subsequent runs** will be much faster (2-3 minutes)
- **Pipeline execution** typically takes 3-5 minutes for the complete dataset
- **All data is persisted** in Docker volumes between restarts
- **Models are saved both locally and in MLflow** (with fallback handling)

---

**🎉 Congratulations!** You now have a fully functional MLOps pipeline running locally with Airflow orchestration, MLflow model tracking, and Redis data caching.
