"""
main.py - Entry point.
Run: python main.py
"""

from src.logger import Logger
import sys
from src.exception import CustomException
from src.pipeline.train_pipeline import TrainPipeline

# ── Logger setup ──────────────────────────────────────────────
_logger_obj = Logger("main")
logger = _logger_obj.get_logger()

if __name__ == "__main__":
    try:
        logger.info('Pipeline Started')
        
        pipeline = TrainPipeline()
        best_model = pipeline.run()
        
        logger.info(f'Best Model : {best_model}')
        logger.info('Pipeline Completed')
    
    except Exception as e:
        raise CustomException (e,sys)
    