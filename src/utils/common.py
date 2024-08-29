# Import necessary modules
from src.logging.logger import logging
from src.config.configuration import TrainingConfig
import os
import torch

# Create an instance of TrainingConfig
config = TrainingConfig()

def save_checkpoint(state, model_dir, filename="my_checkpoint.pth"):
    """
    Save a checkpoint of the model's state.

    Args:
        state (dict): The state of the model to be saved.
        model_dir (str): Directory where the checkpoint will be saved.
        filename (str, optional): Name of the checkpoint file. Defaults to "my_checkpoint.pth".
    """
    logging.info(f"Saving checkpoint to {model_dir}")
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
    torch.save(state, os.path.join(model_dir, filename))

def load_checkpoint(model, optimizer, model_dir):
    """
    Load a checkpoint and update the model and optimizer states.

    Args:
        model (torch.nn.Module): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to update.
        model_dir (str): Directory where the checkpoint is stored.
        filename (str, optional): Name of the checkpoint file. Defaults to "my_checkpoint.pth".

    Returns:
        tuple: Updated model and optimizer.
    """
    logging.info(f"Loading checkpoint from {model_dir}")
    model_latest = os.listdir(model_dir)[-1]
    checkpoint = torch.load(os.path.join(model_dir, model_latest))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    logging.info(f"Checkpoint loaded successfully from {model_dir}")
    return model, optimizer, best_val_loss

