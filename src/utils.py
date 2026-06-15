"""
utils.py - Shared utility functions for Titanic Survival Prediction.
All reusable logic lives here: saving/loading artifacts, model evaluation.

"""
import sys
import os
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.base  import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


from src.exception import CustomException
from src.logger import Logger


# ----------------logger Setup--------------

_logger_obj = Logger('utils')
logger = _logger_obj.get_logger()

class Utils:
    """
    Utility class for common ML pipeline operations.
    Handles artifact persistence and model evaluation.
    """

    # ── Artifact Handling ──────────────────────────────────────
    
    def save_object(self, file_path:str, obj:Any):
        
        """
        Serialize and save a Python object to disk using pickle.

        Args:
            file_path (str): Destination path for the .pkl file.
            obj (Any):       Object to serialize (model, encoder, etc.)

        Raises:
            CustomException: If saving fails for any reason.
        """
        try:
            save_path = Path(file_path)
            save_path.parent.mkdir(parents= True, exist_ok= True)
            
            with open(file_path,'wb') as f:
                pickle.dump(obj, f)
            
            logger.info(f'Object successfully saved ->{save_path}')
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def load_object(self,file_path:str)-> Any:
        """
        Load and deserialize a pickle object from disk.

        Args:
            file_path (str): Path to the .pkl file.

        Returns:
            Any: The deserialized Python object.

        Raises:
            FileNotFoundError: If the file does not exist.
            CustomException:   If loading fails for any reason.
        """
        try:
            load_path = Path(file_path)
            
            if not load_path.exists():
                raise FileNotFoundError(f'No file Found: {load_path}')
            
            with open(file_path,'rb') as f:
                obj = pickle.load(f)
                
            logger.info(f'Object Loaded successfully: {load_path}')
            return obj
        except FileNotFoundError as e:
            logger.error(str(e))
            raise CustomException(e)
        
        except Exception  as e:
            logger.error(f'Failed to LOAD OBJ : {file_path}:{e}')
            raise CustomException(e, sys)
        
# ===================Model Evaluation =======================================================
    def evaluate_models(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        models: Dict[str, BaseEstimator],
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate multiple sklearn-compatible models.
        Reports Accuracy, F1 Score, and ROC-AUC for each model.
        """
        try:
            report: Dict[str, Dict[str, float]] = {}

            for name, model in models.items():
                logger.info(f'Training model: {name}')

                # Train
                model.fit(x_train, y_train)

                # Predict
                y_pred = model.predict(x_test)

                # Probabilities for ROC-AUC
                try:
                    y_prob = model.predict_proba(x_test)[:, 1]
                    auc = round(roc_auc_score(y_test, y_prob), 4)
                except AttributeError:
                    auc = None
                    logger.warning(f"{name} does not support predict_proba. ROC-AUC skipped.")

                # Metrics — outside except, inside for loop
                acc = round(accuracy_score(y_test, y_pred), 4)
                f1  = round(f1_score(y_test, y_pred), 4)

                report[name] = {
                    'accuracy': acc,   # ← no space
                    'f1_score': f1,
                    'roc_auc': auc
                }

                logger.info(f'{name} -> Accuracy: {acc} | F1: {f1} | ROC-AUC: {auc}')

            return report

        except Exception as e:
            logger.error(f'Model evaluation failed: {e}')
            raise CustomException(e)
        