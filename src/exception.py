import sys
from src.logger import logger 

class CustomException(Exception):
    
    def __init__(self, error_message, error_detail:sys)-> str:
        super().__init__(error_message)
        self.error_message = self._get_detailed_error(error_message, error_detail)
        logger.error(self.error_message)
        
    @staticmethod    
    def _get_detailed_error(error_message, error_detail:sys):
        _,_,exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        return(
            f'error occured in File {file_name}| line no {line_number}| error message { str(error_message)}'
        )
    
    def __str__(self):
        return self.error_message