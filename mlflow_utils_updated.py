import os
import mlflow
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.base import BaseEstimator
from datetime import datetime

class MLFlowTracker:
    def __init__(self, config_path: str = "/opt/airflow/config/config.yaml"):
        """Initialize MLflow tracker with configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.mlflow_config = self.config['mlflow']
        
        # Use environment variables for MLflow configuration
        # These should be set in docker-compose.yml
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", self.mlflow_config['tracking_uri'])
        
        # Use the absolute path for artifact storage
        self.artifact_location = "/opt/airflow/mlflow_artifacts"
        
        # Ensure the artifact directory exists with proper permissions
        os.makedirs(self.artifact_location, exist_ok=True)
        print(f"Artifact location: {self.artifact_location}")
        print(f"Artifact location exists: {os.path.exists(self.artifact_location)}")
        print(f"Artifact location permissions: {oct(os.stat(self.artifact_location).st_mode)[-3:]}")
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        self.experiment_name = self.mlflow_config['experiment_name']
        self.experiment = self._get_or_create_experiment()
        
        self.model_name = self.mlflow_config['model_name']
        
    def _get_or_create_experiment(self) -> str:
        """Get or create an MLflow experiment.
        
        Returns:
            str: Experiment ID
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            # Explicitly set the artifact location when creating the experiment
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=f"file://{self.artifact_location}"
            )
            print(f"Created new experiment '{self.experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment '{self.experiment_name}' with ID: {experiment_id}")
            print(f"Experiment artifact location: {experiment.artifact_location}")
        
        return experiment_id
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Make sure we're using the correct experiment ID
        active_run = mlflow.start_run(experiment_id=self.experiment, run_name=run_name)
        print(f"Started MLflow run: {run_name}")
        print(f"Run ID: {active_run.info.run_id}")
        print(f"Run artifact URI: {mlflow.get_artifact_uri()}")
        
        # Log the configuration
        mlflow.log_params({
            "model_algorithm": self.config['model']['algorithm'],
            **{f"hp_{k}": v for k, v in self.config['model']['hyperparameters'].items()}
        })
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        print("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
        print(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        mlflow.log_metrics(metrics)
        print(f"Logged {len(metrics)} metrics")
    
    def log_model(self, model: BaseEstimator, artifact_path: str = "model") -> None:
        """Log a model to the current run.
        
        Args:
            model: Trained model to log
            artifact_path: Path within the artifact store
        """
        # First save the model locally to a temporary directory
        import tempfile
        import joblib
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, "model.joblib")
        
        try:
            # Save the model locally
            joblib.dump(model, temp_model_path)
            print(f"Saved model temporarily to: {temp_model_path}")
            
            # Log the artifact manually
            mlflow.log_artifact(temp_model_path, artifact_path)
            print(f"Logged model artifact from {temp_model_path} to {artifact_path}")
            
            # Only register the model if needed
            if self.mlflow_config['register_model']:
                # Get the artifact URI for the current run
                run_id = mlflow.active_run().info.run_id
                artifact_uri = mlflow.get_artifact_uri(artifact_path)
                
                # Register the model
                model_details = mlflow.register_model(
                    model_uri=artifact_uri, 
                    name=self.model_name
                )
                print(f"Registered model: {model_details.name} version {model_details.version}")
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    def log_artifact(self, local_path: str) -> None:
        """Log an artifact to the current run.
        
        Args:
            local_path: Path to the artifact file
        """
        mlflow.log_artifact(local_path)
        print(f"Logged artifact: {local_path}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Name of the artifact file
        """
        mlflow.log_dict(dictionary, artifact_file)
        print(f"Logged dictionary to: {artifact_file}")
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """Log a matplotlib figure as an artifact.
        
        Args:
            figure: Matplotlib figure to log
            artifact_file: Name of the artifact file
        """
        mlflow.log_figure(figure, artifact_file)
        print(f"Logged figure to: {artifact_file}")
    
    def get_best_model(self) -> Optional[BaseEstimator]:
        """Get the best model from the registry based on a metric.
        
        Returns:
            The best model or None if no models are found
        """
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions of the model
        model_versions = client.search_model_versions(f"name='{self.model_name}'")
        
        if not model_versions:
            print(f"No versions found for model '{self.model_name}'")
            return None
        
        # Get the latest version
        latest_version = max([int(mv.version) for mv in model_versions])
        model_uri = f"models:/{self.model_name}/{latest_version}"
        
        print(f"Loading best model: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
