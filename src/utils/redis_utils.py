import redis
import pandas as pd
import yaml
import json
import numpy as np
from typing import Dict, Any, Optional, List, Union

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class RedisClient:
    def __init__(self, config_path: str = "/opt/airflow/config/config.yaml"):
        """Initialize Redis client with configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.redis_config = self.config['redis']
        self.client = redis.Redis(
            host=self.redis_config['host'],
            port=self.redis_config['port'],
            db=self.redis_config['db']
        )
        
    def store_dataframe(self, df: pd.DataFrame, key: str) -> bool:
        """Store a pandas DataFrame in Redis.
        
        Args:
            df: DataFrame to store
            key: Redis key to store the data under
            
        Returns:
            bool: Success or failure
        """
        try:
            # Convert DataFrame to JSON string
            json_data = df.to_json(orient='records')
            # Store in Redis
            self.client.set(key, json_data)
            print(f"Stored DataFrame with shape {df.shape} at key '{key}'")
            return True
        except Exception as e:
            print(f"Error storing DataFrame in Redis: {e}")
            return False
    
    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a pandas DataFrame from Redis.
        
        Args:
            key: Redis key where the data is stored
            
        Returns:
            DataFrame or None if key doesn't exist
        """
        try:
            # Get data from Redis
            json_data = self.client.get(key)
            if json_data is None:
                print(f"No data found for key '{key}'")
                return None
            
            # Convert bytes to string before using read_json
            json_str = json_data.decode('utf-8')
            
            # Convert JSON to DataFrame
            df = pd.read_json(json_str, orient='records')
            print(f"Retrieved DataFrame with shape {df.shape} from key '{key}'")
            return df
        except Exception as e:
            print(f"Error retrieving DataFrame from Redis: {e}")
            return None
    
    def store_dict(self, data: Dict[str, Any], key: str) -> bool:
        """Store a dictionary in Redis.
        
        Args:
            data: Dictionary to store
            key: Redis key to store the data under
            
        Returns:
            bool: Success or failure
        """
        try:
            # Convert dict to JSON string using custom encoder for numpy types
            json_data = json.dumps(data, cls=NumpyEncoder)
            # Store in Redis
            self.client.set(key, json_data)
            print(f"Stored dictionary at key '{key}'")
            return True
        except Exception as e:
            print(f"Error storing dictionary in Redis: {e}")
            return False
    
    def get_dict(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a dictionary from Redis.
        
        Args:
            key: Redis key where the data is stored
            
        Returns:
            Dict or None if key doesn't exist
        """
        try:
            # Get data from Redis
            json_data = self.client.get(key)
            if json_data is None:
                print(f"No data found for key '{key}'")
                return None
            
            # Convert bytes to string before using json.loads
            json_str = json_data.decode('utf-8')
            
            # Convert JSON to dict
            data = json.loads(json_str)
            print(f"Retrieved dictionary from key '{key}'")
            return data
        except Exception as e:
            print(f"Error retrieving dictionary from Redis: {e}")
            return None
    
    def delete_key(self, key: str) -> bool:
        """Delete a key from Redis.
        
        Args:
            key: Redis key to delete
            
        Returns:
            bool: Success or failure
        """
        try:
            self.client.delete(key)
            print(f"Deleted key '{key}'")
            return True
        except Exception as e:
            print(f"Error deleting key '{key}': {e}")
            return False
    
    def key_exists(self, key: str) -> bool:
        """Check if a key exists in Redis.
        
        Args:
            key: Redis key to check
            
        Returns:
            bool: True if key exists, False otherwise
        """
        return bool(self.client.exists(key))
