"""
data_ingestion.py - Fetches raw data, splits into train/test,
and saves artifacts for the Titanic Survival Prediction pipeline.

"""

import yaml
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pathlib import Path



from src.exception import CustomException
from src.logger import Logger
from src.utils import Utils

#Logger setup
_logger_obj = Logger('data_ingestion')
logger = _logger_obj.get_logger()

# config Loader
def load_config(config_path:str = 'config/config.yaml')->dict:
    """Load YAML config file and return as dictionary."""
    try:
        with open(config_path, 'r') as f:
            config= yaml.safe_load(f)
        logger.info(f'config loaded from : {config_path}')
        return config
    except Exception as e:
        raise CustomException(e)
    
    
# DataIngestionConfig
    
@dataclass
class DataIngestionConfig:
    raw_data_path:str
    train_data_path:str
    test_data_path:str
    source_url:str
    
# DataIngestion class
class DataIngestion:
    """
    Handles fetching, storing, and splitting the Titanic dataset.

    Flow:
        fetch raw data → save raw → split → save train/test → return paths
    """
    
    def __init__(self):
        try:
            config=load_config()
            di_config = config['data_ingestion']
            
            self.ingestion_config = DataIngestionConfig(
                raw_data_path =di_config['raw_data_path'],
                train_data_path = di_config['train_data_path'],
                test_data_path = di_config['test_data_path'],
                source_url = di_config['source_url']
            )
            
            logger.info('DataIngestion INitialized with config data')
            
        except Exception as e:
            logger.error(f'DataIngestion init failed:{e}')
            raise CustomException(e)
        
    def initiate_data_ingestion(self):
        """
        Main method to run the full data ingestion pipeline.

        Returns:
            tuple: (train_data_path, test_data_path)

        Raises:
            CustomException: If any step fails.
        """
        
        logger.info("═" * 50)
        logger.info("Data Ingestion Started")
        
        try:
            #step1: load raw data
            df = pd.read_csv(self.ingestion_config.source_url)
            logger.info(f'shape of df: {df.shape}')
            logger.info(f'column list{list(df.columns)}')
            
            #step2 : save raw data 
            
            raw_path = Path(self.ingestion_config.raw_data_path)
            raw_path.parent.mkdir(parents = True, exist_ok= True)
            df.to_csv(raw_path,index= False)
            logger.info(f'raw path : {raw_path}')
            
            ##Step 3 Train and Test Split 
            logger.info('Train and test the data')
            
            train_df, test_df = train_test_split(
                df, test_size = 0.2, random_state = 42, stratify = df['Survived'] #maintainiing class balance 
            )
            logger.info(f'Train Shape {train_df.shape} | Test_Shape {test_df.shape}')
            logger.info(
                f"Train Survived ratio: {train_df['Survived'].mean():.2f} | "
                f"Test Survived ratio: {test_df['Survived'].mean():.2f}"
            )
            
            # 4. Save Train and test data 
            train_path = Path(self.ingestion_config.train_data_path)
            test_path = Path(self.ingestion_config.test_data_path)
            
            
            # cheecking path availability
            
            train_path.parent.mkdir(parents= True, exist_ok = True)
            test_path.parent.mkdir(parents = True, exist_ok = True)
            
            train_df.to_csv(train_path, index= False)
            test_df.to_csv(test_path, index= False)
            
            logger.info('data ingestion  Completed')
            logger.info("="*50)
            
            return(
                str(self.ingestion_config.train_data_path),
                str(self.ingestion_config.test_data_path)
            )
        
        except Exception as e:
            logger.error(f'Data Ingestion  Failed {e}')
            raise CustomException(e)
    