"""
logger.py - Centralized logger for Titanic Survival Prediction.
Every file imports Logger class and creates its own named logger.
"""

import os
import logging
from datetime import datetime

class Logger:
    
    """
    Industry-standard class-based logger.
    Creates timestamped log files in logs/ directory.
    Each component gets its own named logger for easy tracing.

    Usage:
        _logger_obj = Logger("data_ingestion")
        logger = _logger_obj.get_logger()
    """

    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_path = os.path.join(os.getcwd(),'logs')
    
    
    def __init__(self,name:str):
        """
        Args:
            name (str): Name of the component using this logger.
                        Example: "data_ingestion", "utils", "main"
        """
        self.name = name
        self._setup_log_directory()
        
    def _setup_log_directory(self):
        ''' creates log /directory if it doesn't exist '''
        os.makedirs(self.logs_path, exist_ok = True)
        
    
    def get_logger(self):
        """
        Returns a configured logger instance for the component.

        Returns:
            logging.Logger: Ready-to-use logger object.
        """
        LOG_FILE_PATH= os.path.join(self.logs_path, self.LOG_FILE)
        
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        #avoid  duplicate handlers if logger already exists 
        if not logger.handlers:
            #file_handler - saves logs to file 
            
            file_handler = logging.FileHandler(LOG_FILE_PATH)
            file_handler.setLevel(logging.INFO)
            
            #- console handler preints logs to terminal 
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.stream.reconfigure(encoding='utf-8')
            
            
            #---Format-----------------------------
            
            formatter = logging.Formatter(
                "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger 

