from src.logger import Logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion

_logger_obj = Logger('main')
logger = _logger_obj.get_logger()


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

di = DataIngestion()
train_path, test_path = di.initiate_data_ingestion()
logger.info("Stage 1 Done")

logger.info("Stage 2: Data Transformation")
dt = DataTransformation()
train_array, test_array, preprocessor_path = dt.initiate_data_transformation(
train_path, test_path)
logger.info("Stage 2 Done")