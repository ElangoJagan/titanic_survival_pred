"""
model_trainer.py - Trains multiple classification models,
evaluates them, and saves the best performing model.

"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from src.logger import Logger
from src.exception import CustomException
from src.utils import Utils 


#---------------> lOGGER OBJ
_logger_obj = Logger('model.trainer')
logger = _logger_obj.get_logger()

# ---------------> Config Loaders
def load_config(config_path = 'config/config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info('Loaded config File')
        return config
    except Exception as e:
        raise CustomException(e,sys)
    
def load_params(params_path:str= 'config/params.yaml'):
    try:
        with open(params_path, 'r')as f:
            params = yaml.safe_load(f)
        logger.info('Loaded Params - Hyperparameter values')
        return params
    except Exception as e:
        raise CustomException(e,sys)
    

#------->Model Trainer Config
@dataclass
class ModelTrainerConfig:
    trained_model_path:str
    model_report_path:str
    base_accuracy:float
    
#-------> Model Trainer Class
class ModelTrainer:
    """
    Trains all models defined in params.yaml,
    evaluates them, and saves the best one.
    
    """
    def __init__(self):
        try:
            config = load_config()
            mt_config = config['model_trainer']
            
            self.trainer_config = ModelTrainerConfig(
                trained_model_path = mt_config['trained_model_config'],
                model_report_path = mt_config['model_report_path'],
                base_accuracy= mt_config['base_accuracy']
            )
            self.utils= Utils()
            logger.info('ModelTrainer initialized with config')
        
        except Exception as e:
            logger.error(f'ModelTrainer init failed: {e}')
            raise CustomException(e,sys)
        
# Build Model ----------->
def _get_models(self,params:dict)->dict:
    
    """
        Builds all model instances with hyperparameters from params.yaml.

        Args:
            params: Dictionary of model hyperparameters

        Returns:
            dict: {model_name: model_instance}
    """
    try:
        models={
            'LogisticRegression':LogisticRegression(**params['LogisticRegression']),
            'DecisionTree': DecisionTreeClassifier(**params['DecisionTree']),
            'RandomForest':RandomForestClassifier(**params['RandomForest']),
            'GradientBoosting':GradientBoostingClassifier(**params['RandomForest']),
            'AdaBoost': AdaBoostClassifier(**params['AdaBoost']),
            'XGBoost': XGBClassifier(**params['XGBoost']),
            'CatBoost':CatBoostClassifier(**params['CatBoost']),
            'SVM':SVC(**params('SVM')),
            'KNN':KNeighborsClassifier(**params('KNN')),
        }
        
        logger.info(f'Models Build: {list(models.keys())}')
        return models
    except Exception as e:
        raise CustomException(e,sys)

#-------> MainMethod: 
def initiate_model_trainer(self, train_array:np.ndarray, test_array:np.ndarray)->str:
        """
        Trains all models, evaluates, saves best model.

        Args:
            train_array: Combined train features + target
            test_array : Combined test features + target

        Returns:
            str: Name of the best model

        Raises:
            CustomException: If training or evaluation fails.
        """
        logger.info("=" * 50)
        logger.info("Model Training Started")
        
        try:
            #step 1 : SPLIT ARRAYS INTO X AND Y 
            x_train = train_array[:,:-1], # all cols except last
            y_train = train_array[:, -1]# last column is target
            x_test = test_array[:,:-1]
            y_test = test_array[:,-1]
            logger.info(f'x_train:{x_train.shape} | x_test: {x_test.shape}')
            
            #Step 2 : Load Params and build models
            params = load_params()
            models = self._get_models(params)
            
            
            #Step3 -----> Evaluating models
            logger.info('Evaluating All models')
            report = self.utils.evaluate_models(
                x_train = x_train, 
                y_train = y_train, 
                x_test= x_test, 
                y_test = y_test,
                models = models
            )
            
            #step 4 - Printing full report
            logger.info('=' * 50)
            logger.info('Model Evoluation Report')
            for model_name, scores in report.items():
                logger.info (
                    f"{model_name:20s} | "
                    f"Accuracy: {scores['accuracy']} | "
                    f"F1: {scores['f1_score']} | "
                    f"ROC-AUC: {scores['roc_auc']}"
                )
            logger.info('=' * 50)
            
            #Step 5: Find Best Model by accuracy 
            best_model_name = max(report,key = lambda x:report[x]['accuracy'])
            best_accuracy =report[best_model_name]['accuracy']
            logger.info(f'best model : {best_model_name} | Accuracy :{best_accuracy}')
            
            #Step6 ---> check against base accuracy 
            if best_accuracy < self.trainer_config.base_accuracy:
                raise Exception(
                    f"No model crossed base accuracy of "
                    f"{self.trainer_config.base_accuracy}. "
                    f"Best was {best_model_name} at {best_accuracy}"
                )
                
            # Step 7 : save best models
            best_model = models[best_model_name]
            self.utils.save_object(
                file_path=self.trainer_config.trained_model_path,
                obj=best_model
            )
            logger.info(f"Best model saved -> {self.trainer_config.trained_model_path}")

            # Step 8: Save full report as JSON
            report_path = Path(self.trainer_config.model_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Model report saved -> {report_path}")

            logger.info("Model Training Completed")
            logger.info("=" * 50)

            return best_model_name

        except Exception as e:
            logger.error(f"Model Training Failed: {e}")
            raise CustomException(e)
            
            