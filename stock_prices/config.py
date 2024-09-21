import torch
import torchmetrics

DEVICE = None


def get_wandb_config(epochs):
    return {
        'architecture': 'LSTM',
        'dataset': 'Google stocks',
        'epochs': epochs,
    }


def get_device():
    global DEVICE
    if DEVICE is None:
        if torch.backends.mps.is_available():
            DEVICE = torch.device('mps')  # Apple Silicon (Metal Performance Shaders)
        elif torch.cuda.is_available():
            DEVICE = torch.device('cuda')  # GPU
        else:
            DEVICE = torch.device('cpu')  # CPU
    return DEVICE


