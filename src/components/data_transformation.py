import os
import sys
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from typing import Tuple, List
from PIL import Image
from src.config.configuration import DataTransformationConfig
from src.exceptions.custom_exceptions import CustomException
from src.logging.logger import logging


class CustomTransform():
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        """
        Initialize the transformation pipeline.
        Args:
            resize (tuple): Desired size of the output image (width, height).
            mean (tuple): Mean for normalization (e.g., ImageNet mean).
            std (tuple): Standard deviation for normalization (e.g., ImageNet std).
        """
        self.resize = config.reshape
        self.mean = config.mean if config.mean else [0.485, 0.456, 0.406]
        self.std = config.std if config.std else [0.229, 0.224, 0.225]
        
        # Define the transformations pipeline
        self.transform_pipeline = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, img):
        """
        Apply the transformations to the image.
        Args:
            img (PIL Image or ndarray): Input image to transform.
        Returns:
            torch.Tensor: Transformed image.
        """
        return self.transform_pipeline(img)
    
class CustomDataset(Dataset):
    def __init__(self, data_dir, save_transformed_dir=None):
        """
        Args:
            data_dir (str): Path to the directory containing the data.
            transform (callable, optional): Optional transform to be applied on a sample.
            save_transformed_dir (str, optional): Directory where the transformed images will be saved.
        """
        self.data_dir = data_dir
        self.transform = CustomTransform()
        self.save_transformed_dir = save_transformed_dir
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

        # Create directory for saving transformed images if it doesn't exist
        if self.save_transformed_dir and not os.path.exists(self.save_transformed_dir):
            os.makedirs(self.save_transformed_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)
        # In this example, the label is derived from the filename; you may adjust as per your dataset.
        label = self.image_files[idx].split('_')[0]  # Example: filename "cat_001.jpg" -> label "cat"
        return image, label
    
    def transform_save(self, split: float = 0.2):
        """
        Apply the transformations to the data, and save the transformed images. 
        Args:
            split (float): Percentage of the data to be used for validation.
        """
        try:
            for idx in range(len(self)):
                img, label = self.__getitem__(idx)
                img = self.transform(img)
                out = lambda x : 1 if random.random() < x else 0
                if out(split):
                    img.save(os.path.join(self.save_transformed_dir + '/val', f'{label}_{idx}.{DataTransformationConfig.file_type}'))
                else:
                    img.save(os.path.join(self.save_transformed_dir + '/train', f'{label}_{idx}.{DataTransformationConfig.file_type}'))
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)