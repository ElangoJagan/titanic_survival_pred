from src.logger import Logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion

_logger_obj = Logger('main')
logger = _logger_obj.get_logger()


if __name__ == "__main__":
    try:
        logger.info("Pipeline Started")

        # Stage 1 - Data Ingestion
        logger.info("Stage 1: Data Ingestion")
        di = DataIngestion()
        train_path, test_path = di.initiate_data_ingestion()
        logger.info(f"Train → {train_path}")
        logger.info(f"Test  → {test_path}")
        logger.info("Stage 1 Done ✅")

        logger.info("Pipeline Completed ✅")

    except CustomException as e:
        logger.error(f"Pipeline Failed: {e}")