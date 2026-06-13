"""
data_transformation.py - Feature engineering, encoding, scaling
for Titanic Survival Prediction pipeline.

"""
import numpy as np
import pandas as pd
import os
import sys
import yaml
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.logger import Logger
from src.exception import CustomException
from src.utils import Utils

# Logging Obj creation---------------------------------------------------------
_logger_obj= Logger('Data_Transformation')
logger = _logger_obj.get_logger()

#Config loader

def load_config(config_path:str='config/config.yaml')-> dict:
    try:
        with open(config_path, 'r') as f:
            config= yaml.safe_load(f)
        logger.info(f'config loaded from : {config_path}')
        return config
    except Exception as e:
        logger.error(f'failed to load config: {e}')
        raise CustomException(e,sys)

# DataTransofrmation Config: 
@dataclass            
class DataTransformationConfig:
    preprocessor_path:str
    
# data Transformation Class

class DataTransformation:
    """
    Handles all feature engineering, encoding and scaling.

    Flow:
        load data → feature engineering → build preprocessor
        → fit on train → transform train/test → save preprocessor
        
    """
    def __init__(self):
        try:
            config = load_config()
            dt_config = config['data_transformation']
            self.transformation_config = DataTransformationConfig(preprocessor_path = dt_config['preprocessor_path'])
            
            self.utils = Utils()
            logger.info('DataTransformation Initialized with config')
        except Exception as e:
            raise CustomException(e,sys)
        
        #Feature Engineering
    
    def _feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Creates new features from existing columns.

        New Features:
            FamilySize : Total family members onboard
            IsAlone    : 1 if passenger is alone, else 0
            Title      : Social title extracted from Name

        Args:
            df: Raw dataframe

        Returns:
            df: Dataframe with new features added
        """
        try:
            df= df.copy()
            
            #Family Size = total Family including self
            df['FamilySize'] = df['SibSp']+df['Parch']+1
            logger.info('Feature Created : FamilySize')
            
            #IsAlone orNot
            df['IsAlone']= (df['FamilySize']==1).astype(int)
            logger.info('updating the ISAlone is 1')
            
            #Title:
            df['Title']= df['Name'].str.extract(r'([A-Za-z]+)\.',expand = False)
            
            #Group rare titles
            rare_titles = [
                "Dr", "Rev", "Col", "Major", "Mlle",
                "Countess", "Capt", "Jonkheer", "Lady",
                "Sir", "Mme", "Don"
            ]
            
            df['Title'] = df['Title'].replace(rare_titles, 'Rare')
            df['Title']= df['Title'].replace({'Ms':'Miss','Mlle':'Miss'})
            logger.info(f'feature created : Title | Unique Titles: {df['Title'].unique()}')
            
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    # -- Build PREPROCESSOR - ---------
    def _build_preprocessor(self) -> ColumnTransformer:
        
        """
        Builds sklearn pipeline for numerical and categorical columns.

        Numerical pipeline  : fill missing with median → scale
        Categorical pipeline: fill missing with most frequent → encode

        Returns:
            ColumnTransformer: Fitted preprocessor object
        """
        try:
            #columns to use after feature engineering
            numerical_cols = ['Pclass', 'Age', 'Fare', 'FamilySize', 'IsAlone']
            categorical_cols = ['Sex', 'Embarked','Title']
            
            #numericalPipeline
            num_pipeline = Pipeline(steps = [
                ('imputer',SimpleImputer(strategy= 'median')),
                ('scaler',StandardScaler())
            ])
            
            
            # ------> Categorical Pipeline
            cat_pipeline = Pipeline( steps= [
                ('imputer', SimpleImputer(strategy= 'most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            ## ------> Combine both Pipelines
            preprocessor = ColumnTransformer(transformers = [
                ('num', num_pipeline, numerical_cols), 
                ('cat', cat_pipeline, categorical_cols)
            ])
            
            logger.info(f'numerical colummns : {numerical_cols}')
            logger.info(f'categorical cols:{categorical_cols}')
            logger.info('Pipeline process successfully Completed')
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    #------Main Method
    def initiate_data_transformation(self, train_path:str, test_path:str):
        
        """
        Runs full transformation pipeline on train and test data.

        Args:
            train_path: Path to train CSV
            test_path : Path to test CSV

        Returns:
            tuple: (train_array, test_array, preprocessor_path)

        Raises:
            CustomException: If any step fails.
        """
        logger.info("═" * 50)
        logger.info("Data Transformation Started")
        
        try:
            #step -1 :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f'Train shape : {train_df.shape} | Test Shape :{ test_df.shape}')
            
            
            #Step -2:
            logger.info('Starting Feature Engineering')
            train_df = self._feature_engineering(train_df)
            test_df = self._feature_engineering(test_df)
            logger.info('Feature Engineering Completed')
            
            
            #sTEP -3 
            drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]
            train_df = train_df.drop(columns = drop_cols)
            test_df = test_df.drop(columns = drop_cols)
            logger.info(f'Dropper Columns : {drop_cols}')
            
            #Step 4: Separate features and target
            target = 'Survived'
            x_train = train_df.drop(columns = target)
            y_train= train_df[target]
            x_test = test_df.drop(columns = target)
            y_test = test_df[target]
            logger.info(f"x_train: {x_train.shape} | x_test: {x_test.shape}")
            
            
            # Step 5 ---------> Build & Fit Preprocessor
            preprocessor = self._build_preprocessor()
            
            #FIT on Train only - never fit on test 
            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed = preprocessor.transform(x_test)
            logger.info('Preprocessing fit on train, transform applied to both')
            
            #Step - 6:  Combine features + target intoarrays
            
            train_array = np.c_[x_train_transformed, np.array(y_train)]
            test_array = np.c_[x_test_transformed, np.array(y_test)]
            logger.info(f'Train array shape: {train_array.shape}')
            logger.info(f'test array shape: {test_array.shape}')
            
            # Step 7  save preprocessor:
            self.utils.save_object(file_path = self.transformation_config.preprocessor_path, obj = preprocessor)
            
            logger.info(f"Preprocessor saved → {self.transformation_config.preprocessor_path}")
            logger.info("Data Transformation Completed")
            logger.info("═" * 50)
            
            return(
                train_array,
                test_array,
                self.transformation_config.preprocessor_path
            )
        except Exception as e:
            raise CustomException (e,sys)
            