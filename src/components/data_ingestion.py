import os
import sys
from src.config.configuration import DataConfig
from src.exceptions.custom_exceptions import CustomException
from src.components.data_transformation import DataTransformation
from src.logging.logger import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataIngestionConfig():
    source: str = DataConfig.source
    destination: str = DataConfig.destination
    file_type: str = DataConfig.file_type



class DataIngestion():
    def __init__(self, config: DataIngestionConfig):
        self.source = config.source
        self.destination = config.destination
        self.file_type = config.file_type

    def ingest(self):
        try:
            if not os.path.exists(self.source):
                raise CustomException(f'{self.source} path does not exist', sys)
            
            if not os.path.exists(self.destination):
                raise CustomException(f'{self.destination} path does not exist', sys)
            
            if not self.file_type:
                raise CustomException('File type not specified', sys)

            # these line's read the input images, transforms them and saves them in the destination folder
            DataTransformation(self.source + '/train', self.destination + '/train').transform()
            DataTransformation(self.source + '/val', self.destination + '/val').transform()

        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)