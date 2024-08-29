import os
import torch
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Tuple, List

load_dotenv()

@dataclass
class DataConfig():
    source: str = os.getenv('DATA_SOURCE_PATH')
    destination: str = os.getenv('DATA_DESTINATION_PATH')
    file_type: str = os.getenv('DATA_FILE_TYPE')

@dataclass
class DataTransformationConfig():
    reshape: Tuple[int, int] = (28, 28)
    std: List[int, int, int] = [0.229, 0.224, 0.225]
    mean: List[int, int, int] = [0.485, 0.456, 0.406]
    file_type: str = os.getenv('DATA_FILE_TYPE')

@dataclass
class TrainingConfig():
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_decay: float = 0.0001
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    registry: str = os.getenv('REGISTRY_PATH')