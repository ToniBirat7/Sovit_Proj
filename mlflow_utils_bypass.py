import os
import mlflow
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.base import BaseEstimator
from datetime import datetime
import joblib

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
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", self.mlflow_config['tracking_uri'])
        self.artifact_location = "/opt/airflow/mlflow_artifacts"
        
        # Flag to indicate if MLflow is available
        self.mlflow_available = True
        
        # Ensure the artifact directory exists with proper permissions
        try:
            os.makedirs(self.artifact_location, exist_ok=True)
            print(f"Artifact location: {self.artifact_location}")
            print(f"Artifact location exists: {os.path.exists(self.artifact_location)}")
            print(f"Artifact location permissions: {oct(os.stat(self.artifact_location).st_mode)[-3:]}")
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set experiment
            self.experiment_name = self.mlflow_config['experiment_name']
            self.experiment = self._get_or_create_experiment()
            
            self.model_name = self.mlflow_config['model_name']
            self.active_run = None
        except Exception as e:
            print(f"MLflow initialization failed: {str(e)}")
            self.mlflow_available = False
        
    def _get_or_create_experiment(self) -> str:
        """Get or create an MLflow experiment.
        
        Returns:
            str: Experiment ID
        """
        if not self.mlflow_available:
            return None
            
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=f"file://{self.artifact_location}"
                )
                print(f"Created new experiment '{self.experiment_name}' with ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing experiment '{self.experiment_name}' with ID: {experiment_id}")
            
            return experiment_id
        except Exception as e:
            print(f"Error in _get_or_create_experiment: {str(e)}")
            self.mlflow_available = False
            return None
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
        """
        if not self.mlflow_available:
            print("MLflow is not available. Skipping start_run.")
            return
            
        try:
            if run_name is None:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.active_run = mlflow.start_run(experiment_id=self.experiment, run_name=run_name)
            print(f"Started MLflow run: {run_name}")
            print(f"Run ID: {self.active_run.info.run_id}")
            print(f"Run artifact URI: {mlflow.get_artifact_uri()}")
            
            # Log the configuration
            mlflow.log_params({
                "model_algorithm": self.config['model']['algorithm'],
                **{f"hp_{k}": v for k, v in self.config['model']['hyperparameters'].items()}
            })
        except Exception as e:
            print(f"Error in start_run: {str(e)}")
            self.mlflow_available = False
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.mlflow_available:
            print("MLflow is not available. Skipping end_run.")
            return
            
        try:
            mlflow.end_run()
            print("Ended MLflow run")
        except Exception as e:
            print(f"Error in end_run: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.mlflow_available:
            print(f"MLflow is not available. Saving parameters locally: {params}")
            return
            
        try:
            mlflow.log_params(params)
            print(f"Logged {len(params)} parameters")
        except Exception as e:
            print(f"Error in log_params: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        if not self.mlflow_available:
            print(f"MLflow is not available. Saving metrics locally: {metrics}")
            return
            
        try:
            mlflow.log_metrics(metrics)
            print(f"Logged {len(metrics)} metrics")
        except Exception as e:
            print(f"Error in log_metrics: {str(e)}")
    
    def log_model(self, model: BaseEstimator, artifact_path: str = "model") -> None:
        """Log a model to the current run.
        
        Args:
            model: Trained model to log
            artifact_path: Path within the artifact store
        """
        # Save the model locally regardless of MLflow availability
        os.makedirs("/opt/airflow/models", exist_ok=True)
        model_path = f"/opt/airflow/models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved locally to: {model_path}")
        
        if not self.mlflow_available:
            print("MLflow is not available. Model was only saved locally.")
            return
            
        try:
            # First try to log the model directly to MLflow
            try:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=self.model_name if self.mlflow_config['register_model'] else None
                )
                print(f"Logged model to MLflow path: {artifact_path}")
            except Exception as e:
                print(f"Failed to log model directly to MLflow: {str(e)}")
                
                # If that fails, try a simpler approach - log the saved file as an artifact
                try:
                    mlflow.log_artifact(model_path, artifact_path)
                    print(f"Logged model as artifact to MLflow path: {artifact_path}")
                except Exception as e2:
                    print(f"Failed to log model as artifact: {str(e2)}")
                    self.mlflow_available = False
        except Exception as e:
            print(f"Error in log_model: {str(e)}")
            self.mlflow_available = False
    
    def log_artifact(self, local_path: str) -> None:
        """Log an artifact to the current run.
        
        Args:
            local_path: Path to the artifact file
        """
        if not self.mlflow_available:
            print(f"MLflow is not available. Artifact was not logged: {local_path}")
            return
            
        try:
            mlflow.log_artifact(local_path)
            print(f"Logged artifact: {local_path}")
        except Exception as e:
            print(f"Error in log_artifact: {str(e)}")
            self.mlflow_available = False
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_file: Name of the artifact file
        """
        if not self.mlflow_available:
            print(f"MLflow is not available. Dictionary was not logged as {artifact_file}")
            return
            
        try:
            mlflow.log_dict(dictionary, artifact_file)
            print(f"Logged dictionary to: {artifact_file}")
        except Exception as e:
            print(f"Error in log_dict: {str(e)}")
            self.mlflow_available = False
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """Log a matplotlib figure as an artifact.
        
        Args:
            figure: Matplotlib figure to log
            artifact_file: Name of the artifact file
        """
        if not self.mlflow_available:
            print(f"MLflow is not available. Figure was not logged as {artifact_file}")
            return
            
        try:
            mlflow.log_figure(figure, artifact_file)
            print(f"Logged figure to: {artifact_file}")
        except Exception as e:
            print(f"Error in log_figure: {str(e)}")
            self.mlflow_available = False
    
    def get_best_model(self) -> Optional[BaseEstimator]:
        """Get the best model from the registry based on a metric.
        
        Returns:
            The best model or None if no models are found
        """
        if not self.mlflow_available:
            print("MLflow is not available. Could not get best model from registry.")
            # Try to find the latest locally saved model
            try:
                model_dir = "/opt/airflow/models"
                if os.path.exists(model_dir):
                    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".joblib")]
                    if model_files:
                        latest_model = sorted(model_files)[-1]
                        model_path = os.path.join(model_dir, latest_model)
                        print(f"Loading latest local model: {model_path}")
                        return joblib.load(model_path)
            except Exception as e:
                print(f"Error loading local model: {str(e)}")
            return None
            
        try:
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
        except Exception as e:
            print(f"Error in get_best_model: {str(e)}")
            self.mlflow_available = False
            return None
