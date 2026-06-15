"""
predict_pipeline.py - Loads saved model and preprocessor,
transforms new input data and returns survival prediction.

"""
import sys
import pandas as pd
from src.logger import Logger
from src.exception import CustomException
from src.utils import Utils 

# ------------log
_logger_obj = Logger('PredictPipeline')
logger = _logger_obj.get_logger()

class PredictPipeline:
    """
    Loads saved artifacts and predicts survival for new passengers.
    
    """
    def __init__(self):
        self.utils = Utils()
        logger.info('PredictPipeline Initialized')
        
    def predict(self,input_df:pd.DataFrame)-> str:
        
        """
        Takes raw passenger input, preprocesses and predicts survival.

        Args:
            input_df: Raw passenger details as DataFrame

        Returns:
            str: "Survived" or "Did Not Survive"

        Raises:
            CustomException: If prediction fails.
        """
        try:
            #Step1 : Load Saved Artifacts
            preprocessor = self.utils.load_object('artifacts/data_transformation/preprocess.pkl')
            model = self.utils.load_object('artifacts/models/model/pkl')
            logger.info('Preprocessor and Model Loaded Successfully')
            
            #Step 2: transform Input
            input_transformed = preprocessor.transform(input_df)
            logger.info(f'input transformed {input_transformed.shape}')
            
            #Step3:Predict
            prediction = model.predict(input_transformed)
            logger.info(f'raw prediction: {prediction}')
            
            # Step4: return  human readable result 
            result =  'Survived' if prediction[0] ==1 else 'Did Not survive '
            logger.info('Prediction Result:{result}')
            
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise CustomException(e, sys)
class CustomData:
    """
    Captures raw input from Flask form and converts to DataFrame.
    This is what connects the HTML form to the predict pipeline.
    """
    
    def __init__(
        self,
        pclass: int,
        sex: str,
        age: float,
        fare: float,
        embarked: str,
        sibsp: int,
        parch: int  
    ):
        self.pclass   = pclass
        self.sex      = sex
        self.age      = age
        self.fare     = fare
        self.embarked = embarked
        self.sibsp    = sibsp
        self.parch    = parch
    
    def get_data_as_dataframe(self)->pd.DataFrame:
        """
        Converts form input to DataFrame with feature engineering applied.

        Returns:
            pd.DataFrame: Ready for preprocessor transformation
        """
        try:
            #Feature engineeering -  same as training
            family_size= self.sibsp+ self.parch+1
            is_alone = 1 if family_size==1 else 0
            
            data= {
                'Pclass': [self.pclass],
                'Sex':[self.sex],
                'Age':[self.age],
                'Fare':[self.fare],
                'Embarked':[self.embarked],
                'FamilySize':[family_size],
                'IsAlone':[is_alone],
                'Title':[self.extract_title()],
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Input DataFrame created: {df.shape}")
            logger.info(f"Input data:\n{df}")
            return df
        
        except Exception as e:
            logger.error(f"DataFrame creation failed: {e}")
            raise CustomException(e, sys)
    def _extract_title(self) -> str:
        """
        Returns title based on sex.
        Since we don't have Name in form input,
        we assign title based on sex — industry standard workaround.

        Returns:
            str: Title (Mr/Mrs/Miss)
        """
        if self.sex == "male":
            return "Mr"
        else:
            return "Mrs"