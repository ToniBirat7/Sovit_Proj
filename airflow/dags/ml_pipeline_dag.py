from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
import os
import pandas as pd
import mlflow
import yaml
import joblib

# Add project directory to path
sys.path.insert(0, '/opt/airflow')

# Import project modules
from src.data.cleaning import read_csv_data, clean_data, get_data_summary
from src.data.transformation import transform_data
from src.data.validation import validate_data
from src.features.feature_engineering import engineer_features
from src.models.training import split_data, train_model, grid_search_cv
from src.models.evaluation import evaluate_model
from src.utils.redis_utils import RedisClient
from src.utils.mlflow_utils import MLFlowTracker

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'lung_cancer_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for lung cancer mortality prediction',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'lung_cancer', 'prediction'],
)

# Load configuration
def _load_config():
    with open("/opt/airflow/config/config.yaml", 'r') as file:
        return yaml.safe_load(file)

config = _load_config()

# 1. Data Ingestion Task
def ingest_data(**kwargs):
    """Ingest the raw data from CSV."""
    print("Starting data ingestion task...")
    df = read_csv_data()
    print(f"Data ingested successfully. Shape: {df.shape}")
    
    # Get data summary
    summary = get_data_summary(df)
    print(f"Data summary: {summary}")
    
    # Pass data to the next task via XCom
    kwargs['ti'].xcom_push(key='raw_data_shape', value=df.shape)
    return "Data ingestion completed"

# 2. Data Cleaning Task
def clean_data_task(**kwargs):
    """Clean the raw data."""
    print("Starting data cleaning task...")
    df = read_csv_data()
    df_cleaned = clean_data(df)
    
    # Store in Redis
    redis_client = RedisClient()
    redis_key = config['redis']['keys']['raw_data']
    redis_client.store_dataframe(df_cleaned, redis_key)
    
    # Pass data to the next task via XCom
    kwargs['ti'].xcom_push(key='cleaned_data_shape', value=df_cleaned.shape)
    return "Data cleaning completed"

# 3. Data Transformation Task
def transform_data_task(**kwargs):
    """Transform the cleaned data."""
    print("Starting data transformation task...")
    
    # Get cleaned data from Redis
    redis_client = RedisClient()
    redis_key = config['redis']['keys']['raw_data']
    df_cleaned = redis_client.get_dataframe(redis_key)
    
    if df_cleaned is None:
        raise ValueError("Cleaned data not found in Redis")
    
    # Transform data
    df_transformed, transform_meta = transform_data(df_cleaned)
    
    # Store transformed data in Redis
    transformed_key = config['redis']['keys']['transformed_data']
    redis_client.store_dataframe(df_transformed, transformed_key)
    
    # Store transformation metadata in Redis
    redis_client.store_dict(transform_meta, f"{transformed_key}:meta")
    
    # Pass data to the next task via XCom
    kwargs['ti'].xcom_push(key='transformed_data_shape', value=df_transformed.shape)
    kwargs['ti'].xcom_push(key='transform_meta', value=transform_meta)
    return "Data transformation completed"

# 4. Data Validation Task
def validate_data_task(**kwargs):
    """Validate the transformed data."""
    print("Starting data validation task...")
    
    # Get transformed data from Redis
    redis_client = RedisClient()
    transformed_key = config['redis']['keys']['transformed_data']
    df_transformed = redis_client.get_dataframe(transformed_key)
    
    if df_transformed is None:
        raise ValueError("Transformed data not found in Redis")
    
    # Get transformation metadata from Redis
    transform_meta = redis_client.get_dict(f"{transformed_key}:meta")
    
    if transform_meta is None:
        raise ValueError("Transformation metadata not found in Redis")
    
    # Validate data
    validation_passed, validation_results = validate_data(df_transformed, transform_meta)
    
    if not validation_passed:
        print("WARNING: Data validation failed. Check validation results for details.")
        print(f"Validation warnings: {validation_results['warnings']}")
    
    # Pass validation results to the next task via XCom
    kwargs['ti'].xcom_push(key='validation_passed', value=validation_passed)
    kwargs['ti'].xcom_push(key='validation_results', value=validation_results)
    return "Data validation completed"

# 5. Feature Engineering Task
def feature_engineering_task(**kwargs):
    """Perform feature engineering on the validated data."""
    print("Starting feature engineering task...")
    
    # Get transformed data from Redis
    redis_client = RedisClient()
    transformed_key = config['redis']['keys']['transformed_data']
    df_transformed = redis_client.get_dataframe(transformed_key)
    
    if df_transformed is None:
        raise ValueError("Transformed data not found in Redis")
    
    # Engineer features
    df_engineered, fe_meta = engineer_features(df_transformed)
    
    # Store engineered data in Redis
    model_features_key = config['redis']['keys']['model_features']
    redis_client.store_dataframe(df_engineered, model_features_key)
    
    # Store feature engineering metadata in Redis
    redis_client.store_dict(fe_meta, f"{model_features_key}:meta")
    
    # Pass data to the next task via XCom
    kwargs['ti'].xcom_push(key='engineered_data_shape', value=df_engineered.shape)
    kwargs['ti'].xcom_push(key='feature_engineering_meta', value=fe_meta)
    return "Feature engineering completed"

# 6. Model Training Task
def train_model_task(**kwargs):
    """Train the model on the engineered data."""
    print("Starting model training task...")
    
    # Get engineered data from Redis
    redis_client = RedisClient()
    model_features_key = config['redis']['keys']['model_features']
    df_engineered = redis_client.get_dataframe(model_features_key)
    
    if df_engineered is None:
        raise ValueError("Engineered data not found in Redis")
    
    # Initialize MLflow tracker
    mlflow_tracker = MLFlowTracker()
    mlflow_tracker.start_run(run_name="training_run")
    
    # Split data
    # Assume the target column is the last column
    target_column = df_engineered.columns[-1]
    X_train, y_train, X_test, y_test = split_data(df_engineered, target_column)
    
    # Train model
    model, training_meta = train_model(X_train, y_train)
    
    # Log model and parameters to MLflow
    mlflow_tracker.log_params(training_meta["hyperparameters"])
    mlflow_tracker.log_model(model)
    
    # Store test data in Redis for evaluation
    redis_client.store_dataframe(X_test.assign(**{target_column: y_test.values}), f"{model_features_key}:test")
    
    # Pass data to the next task via XCom
    kwargs['ti'].xcom_push(key='training_meta', value=training_meta)
    kwargs['ti'].xcom_push(key='model_algorithm', value=training_meta["algorithm"])
    
    # End MLflow run
    mlflow_tracker.end_run()
    return "Model training completed"

# 7. Model Evaluation Task
def evaluate_model_task(**kwargs):
    """Evaluate the trained model."""
    print("Starting model evaluation task...")
    
    # Get test data from Redis
    redis_client = RedisClient()
    model_features_key = config['redis']['keys']['model_features']
    test_data = redis_client.get_dataframe(f"{model_features_key}:test")
    
    if test_data is None:
        raise ValueError("Test data not found in Redis")
    
    # Get the model algorithm from XCom
    ti = kwargs['ti']
    model_algorithm = ti.xcom_pull(task_ids='train_model_task', key='model_algorithm')
    
    # Load the trained model
    model_path = f"/opt/airflow/models/{model_algorithm.lower()}_model.joblib"
    model = joblib.load(model_path)
    
    # Initialize MLflow tracker
    mlflow_tracker = MLFlowTracker()
    mlflow_tracker.start_run(run_name="evaluation_run")
    
    # Assume the target column is the last column
    target_column = test_data.columns[-1]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Log metrics to MLflow
    mlflow_tracker.log_metrics(metrics)
    
    # End MLflow run
    mlflow_tracker.end_run()
    
    # Pass metrics to the next task via XCom
    kwargs['ti'].xcom_push(key='evaluation_metrics', value=metrics)
    return "Model evaluation completed"

# 8. Model Registration Task
def register_model_task(**kwargs):
    """Register the model in MLflow model registry."""
    print("Starting model registration task...")
    
    # Get the model algorithm from XCom
    ti = kwargs['ti']
    model_algorithm = ti.xcom_pull(task_ids='train_model_task', key='model_algorithm')
    
    # Initialize MLflow tracker
    mlflow_tracker = MLFlowTracker()
    
    # Get the best model
    best_model = mlflow_tracker.get_best_model()
    
    if best_model is None:
        print("No model found in MLflow registry. Using the trained model.")
        # Load the trained model
        model_path = f"/opt/airflow/models/{model_algorithm.lower()}_model.joblib"
        best_model = joblib.load(model_path)
        
        # Register the model
        mlflow_tracker.start_run(run_name="registration_run")
        mlflow_tracker.log_model(best_model)
        mlflow_tracker.end_run()
    
    return "Model registration completed"

# Create the tasks
ingest_task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_data,
    provide_context=True,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_data_task',
    python_callable=clean_data_task,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data_task',
    python_callable=transform_data_task,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data_task',
    python_callable=validate_data_task,
    provide_context=True,
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering_task',
    python_callable=feature_engineering_task,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=evaluate_model_task,
    provide_context=True,
    dag=dag,
)

register_model_task = PythonOperator(
    task_id='register_model_task',
    python_callable=register_model_task,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
ingest_task >> clean_task >> transform_task >> validate_task >> feature_engineering_task >> train_model_task >> evaluate_model_task >> register_model_task
