# Lung Cancer Mortality Prediction MLOps Pipeline

This project implements an end-to-end MLOps pipeline for lung cancer mortality prediction. The pipeline automates the entire machine learning lifecycle from data ingestion to model deployment using Airflow for orchestration, MLflow for model tracking, and Redis for fast data access.

## Project Structure

```
Sovit_Proj/
├── Dataset/
│   └── lung_cancer_mortality_data_test_v2.csv
├── airflow/
│   ├── dags/
│   │   └── ml_pipeline_dag.py
│   └── plugins/
│       └── operators/
├── src/
│   ├── data/
│   │   ├── cleaning.py
│   │   ├── transformation.py
│   │   └── validation.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── training.py
│   │   └── evaluation.py
│   └── utils/
│       ├── redis_utils.py
│       └── mlflow_utils.py
├── config/
│   └── config.yaml
├── docker-compose.yml
├── requirements.txt
├── RUN.md                    # 📖 Complete setup guide
├── quick_setup.sh           # 🚀 Automated setup script
├── verify_setup.sh          # ✅ Health check script
├── fix_issues.sh            # 🔧 Quick troubleshooting tool
├── README.md
├── setup.sh
├── start_project.sh
└── explore_dataset.py
```

## 📋 Available Scripts

- **`RUN.md`** - Complete step-by-step setup guide with troubleshooting
- **`quick_setup.sh`** - Automated setup script (recommended for first-time users)
- **`verify_setup.sh`** - Health check script to verify all components are working
- **`fix_issues.sh`** - Interactive troubleshooting tool for common issues
- **`setup.sh`** - Manual environment setup
- **`start_project.sh`** - Legacy start script

## 🚀 Quick Start

### For New Users (Recommended)

If you're cloning this repository for the first time, follow the **comprehensive setup guide**:

📖 **[READ THE COMPLETE SETUP GUIDE: RUN.md](./RUN.md)**

The RUN.md file contains detailed step-by-step instructions with troubleshooting for all common issues.

### Automated Setup (One-Click)

For experienced users who want automated setup:

```bash
# Clone the repository
git clone <your-repository-url>
cd <repository-name>

# Run the automated setup script
./quick_setup.sh

# Verify everything is working
./verify_setup.sh
```

### Manual Setup

If you prefer manual control over the setup process:

```bash
git clone <repository-url>
cd Sovit_Proj
```

2. The easiest way to set up the project is to use the automated start script:

```bash
chmod +x start_project.sh
./start_project.sh
```

This script will:

- Check for required dependencies
- Create a Python virtual environment
- Install all dependencies
- Start Docker containers for Airflow, MLflow, and Redis

3. Alternatively, you can set up the environment manually:

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate

# Start Docker containers
docker-compose up -d
```

4. Explore the dataset and update configuration:

```bash
# Activate the virtual environment if not already active
source venv/bin/activate

# Run the dataset exploration script
./explore_dataset.py
```

This will analyze the dataset and update the configuration in `config/config.yaml`.

## Pipeline Workflow

The MLOps pipeline consists of the following steps:

1. **Data Ingestion**: Read the CSV file
2. **Data Cleaning**: Clean the data
3. **Redis Storage (Raw)**: Store cleaned data in Redis
4. **Data Transformation**: Fetch data from Redis and transform
5. **Data Validation**: Validate the transformed data
6. **Redis Storage (Transformed)**: Store transformed data in Redis
7. **Feature Engineering**: Generate new features from the transformed data
8. **Model Training**: Fetch transformed data and train model
9. **Model Evaluation**: Evaluate model performance
10. **Model Registration**: Register model in MLflow

## Airflow UI

Access the Airflow UI at http://localhost:8080 with the following credentials:

- Username: airflow
- Password: airflow

## MLflow UI

Access the MLflow UI at http://localhost:5000 to track experiments, compare models, and manage the model registry.

## Redis

Redis is running on port 6379 and can be accessed using the Redis CLI or any Redis client.

## Configuration

All configuration parameters are stored in `config/config.yaml`. You can modify these parameters to customize the pipeline behavior.

## Monitoring

- **Airflow**: Provides monitoring of pipeline execution
- **MLflow**: Tracks model performance metrics and parameters
- **Redis**: Stores intermediate data for fast access

## Troubleshooting

⚠️ **For detailed troubleshooting, please refer to [RUN.md](./RUN.md)** which contains comprehensive solutions for all common issues.

### Quick Fixes

1. **Run interactive troubleshooting tool**:

   ```bash
   ./fix_issues.sh
   ```

2. **Verify your setup**:

   ```bash
   ./verify_setup.sh
   ```

3. **Check service status**:

```bash
docker-compose ps
```

3. **View service logs**:

```bash
docker-compose logs <service-name>
```

4. **Restart services**:

```bash
docker-compose restart
```

### Common Issues

- **Docker permission denied**: Add user to docker group and restart terminal
- **Port conflicts**: Check if ports 8080, 5000, 6379, 5432 are available
- **Python package errors**: Reinstall packages in Airflow containers
- **DAG not visible**: Wait 2-3 minutes for DAG registration
- **MLflow permission errors**: Pipeline includes fallback to local model storage

📖 **For complete troubleshooting guide, see [RUN.md](./RUN.md)**

## Stopping the Pipeline

To stop all containers:

```bash
docker-compose down
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Apache Airflow
- MLflow
- Redis
- Scikit-learn
- Pandas/NumPy

## 🎯 Expected Results

After successful setup, you should have:

### ✅ Running Services

- **Airflow Web UI**: http://localhost:8080 (admin/admin)
- **MLflow UI**: http://localhost:5000
- **Redis**: Running on port 6379
- **PostgreSQL**: Running on port 5432

### ✅ ML Pipeline Components

The pipeline consists of 8 sequential tasks:

1. **Data Ingestion** - Load raw data from CSV
2. **Data Cleaning** - Preprocess and clean data
3. **Data Transformation** - Apply feature transformations
4. **Data Validation** - Quality checks and validation
5. **Feature Engineering** - Create advanced features
6. **Model Training** - Train RandomForestRegressor
7. **Model Evaluation** - Calculate performance metrics
8. **Model Registration** - Save model artifacts

### ✅ Sample Output

A successful pipeline run produces:

- **Model artifacts** saved in `/opt/airflow/models/`
- **Feature importance analysis** (top feature: `id_to_age_ratio` with 99.5% importance)
- **Performance metrics** in MLflow
- **Training metadata** with hyperparameters and results

### 🚀 Running Your First Pipeline

1. Open Airflow UI at http://localhost:8080
2. Login with username: `admin`, password: `admin`
3. Find the `lung_cancer_ml_pipeline` DAG
4. Toggle it ON (unpause)
5. Click the "Play" button to trigger execution
6. Monitor progress in the Graph View
7. Check results in MLflow UI at http://localhost:5000
