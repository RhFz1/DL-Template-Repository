import sys
from src.components.data_transformation import CustomDataset
from src.config.configuration import DataConfig
from src.logging.logger import logging
from src.exceptions.custom_exceptions import CustomException
from torch.utils.data import DataLoader

class DataLoader():
    def __init__(self, config: DataConfig):
        # Initialize the DataLoader with configuration
        self.config = config

    def load(self, batch_size=32, shuffle=True, num_workers=4, pin_memory=False):
        """
        Load the data and create DataLoader objects for training and validation.

        Args:
            batch_size (int): Number of samples per batch. Default is 32.
            shuffle (bool): Whether to shuffle the data. Default is True.
            num_workers (int): Number of subprocesses to use for data loading. Default is 4.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Default is False.

        Returns:
            tuple: A tuple containing train_loader and val_loader.
        """
        try:
            # Create CustomDataset for training data
            train_data = CustomDataset(data_dir=self.config.destination + '/train')
            # Create DataLoader for training data
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            
            # Create CustomDataset for validation data
            val_data = CustomDataset(data_dir=self.config.destination + '/val')
            # Create DataLoader for validation data
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            
            return train_loader, val_loader
        except Exception as e:
            # Log the exception
            logging.info(e)
            # Raise a custom exception with the caught exception and system information
            raise CustomException(e, sys)