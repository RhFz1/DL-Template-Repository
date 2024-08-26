import os
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataclasses import dataclass
from typing import Tuple, List
from PIL import Image
from src.config.configuration import DataTransformationConfig
from src.exceptions.custom_exceptions import CustomException
from src.logging.logger import logging



@dataclass
class DataTransformationConfig():
    reshape: Tuple[int, int] = DataTransformationConfig.reshape
    std: List[int, int, int] = DataTransformationConfig.std
    mean: List[int, int, int] = DataTransformationConfig.mean


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
        
        if self.transform:
            image = self.transform(image)

            # Save the transformed image if a save directory is provided
            if self.save_transformed_dir:
                save_path = os.path.join(self.save_transformed_dir, self.image_files[idx])
                # Convert the tensor back to an image and save it
                transformed_image = transforms.ToPILImage()(image)
                transformed_image.save(save_path)

        # In this example, the label is derived from the filename; you may adjust as per your dataset.
        label = self.image_files[idx].split('_')[0]  # Example: filename "cat_001.jpg" -> label "cat"

        return image, label
    

class DataTransformation():

    def __init__(self, data_dir: str, save_dir: str = None) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir

    def transform(self, batch_size:int = 4, shuffle: bool = False, num_workers: int = 4, pin_memory: bool = False)-> bool:
        """
        Apply the transformations to the data, and save the transformed images. 
        Args:
            data_dir (str): Path to the directory containing the data.
        Returns:
            DataLoader: Transformed data.
        """

        try: 
            dataset = CustomDataset(self.data_dir, self.save_dir)
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)

    def transform_load(self, batch_size: int = 4, shuffle: bool = False, num_workers: int = 4, pin_memory: bool = False) -> DataLoader:
        """
        Apply the transformations to the data, save the transformed images and also load the transformed data.
        Args:
            data_dir (str): Path to the directory containing the data.
        Returns:
            DataLoader: Transformed data.
        """

        try:
            dataset = CustomDataset(self.data_dir, self.save_dir)
            # the core logic is to return the DataLoader object, which will be used in the training loop.
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)

        return data_loader