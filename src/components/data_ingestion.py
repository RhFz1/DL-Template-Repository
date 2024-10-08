import os
import sys
from src.config.configuration import DataConfig
from src.exceptions.custom_exceptions import CustomException
from src.logging.logger import logging
from src.components.data_transformation import CustomDataset
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
            
            # this line takes in data from the source directory and saves the transformed data in the destination directory with a train val split.
            CustomDataset(data_dir=self.source, save_transformed_dir=self.destination).transform_save(split=0.2)
            
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)