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
├── README.md
├── setup.sh
├── start_project.sh
└── explore_dataset.py
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Git

### Installation

1. Clone the repository:

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

### Installation Issues

1. **Package installation failures**:
   - Make sure your Python version is compatible (Python 3.8+)
   - Try running `pip install --upgrade pip setuptools wheel` before installing other packages
2. **Docker issues**:
   - Ensure Docker and Docker Compose are installed and running
   - Check Docker logs with `docker-compose logs`
3. **Permission issues**:
   - Make sure the script files are executable: `chmod +x *.sh *.py`

### Runtime Issues

1. **Airflow connection issues**:
   - Check if all containers are running: `docker-compose ps`
   - Verify network configuration in `docker-compose.yml`
2. **MLflow tracking errors**:
   - Ensure MLflow container is running
   - Check MLflow logs: `docker-compose logs mlflow`
3. **Redis connection errors**:
   - Verify Redis container is running
   - Check Redis logs: `docker-compose logs redis`

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
