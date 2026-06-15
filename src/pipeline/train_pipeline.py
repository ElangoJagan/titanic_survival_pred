"""
train_pipeline.py - Orchestrates the full training pipeline.
Connects DataIngestion → DataTransformation → ModelTrainer.

"""

import os 
import sys 

from src.logger import Logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

#-------logger
_logger_obj = Logger('Train_Pipeline')
logger = _logger_obj.get_logger()


class TrainPipeline:
    """
    Orchestrates the complete ML training pipeline.

    Flow:
        DataIngestion → DataTransformation → ModelTrainer
        
    """
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
        logger.info('TrainingPipeline initialized')
    
    def run(self):
        
        """
        Runs the complete training pipeline end to end.

        Returns:
            str: Name of the best model found

        Raises:
            CustomException: If any stage fails.
            
        """
        try:
            logger.info("=" * 50)
            logger.info("Training Pipeline Started")
            
            #Stage1 : DataIngestion
            logger.info('Stage1 - DataIngestion Started ')
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logger.info('Stage 1 - Completed')
            
            #Stage 2 : Data Transformation
            logger.info('stage 2 - Data Transformation')
            train_array, test_array, _= self.data_transformation.initiate_data_transformation(train_path, test_path)
            logger.info('Stage2 : Completed')
            
            #Stage 3 : ModelTrainer
            logger.info('Stage 3 - Model Trainer')
            best_model = self.model_trainer.initiate_model_trainer(train_array, test_array)
            logger.info('Stage 3 - Completed')
            logger.info('Training Pipeline COmpleted')
            logger.info('=' * 50)
            
            return best_model
                    
        except Exception  as e:
            raise CustomException(e,sys)