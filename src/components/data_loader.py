import sys
from src.components.data_transformation import CustomDataset
from src.config.configuration import DataConfig
from src.logging.logger import logging
from src.exceptions.custom_exceptions import CustomException
from torch.utils.data import DataLoader

class DataLoader():
    def __init__(self, config: DataConfig):
        self.config = config

    def load(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=False):
        """
        Load the data.
        """
        try:
            train_data = CustomDataset(data_dir=self.config.destination + '/train')
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            val_data = CustomDataset(data_dir=self.config.destination + '/val')
            val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            return train_loader, val_loader
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)