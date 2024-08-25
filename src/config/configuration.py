import os
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