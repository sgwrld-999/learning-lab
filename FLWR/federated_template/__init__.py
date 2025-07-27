"""
Package initialization for the Federated Learning Template

This module provides easy imports for the main components of the
federated learning system.
"""

from .dataset_manager import DatasetManager, CustomDataset
from .model import LocalModel, AdvancedLocalModel, create_model
from .client import FLClient, ClientTrainer
from .main import FederatedLearningRunner, create_sample_datasets
from .config import ConfigManager, ModelConfig, DataConfig, FLConfig, SystemConfig

__version__ = "1.0.0"
__author__ = "Federated Learning Template Team"
__description__ = "Generic Federated Learning Template using FLWR + PyTorch"

__all__ = [
    "DatasetManager",
    "CustomDataset", 
    "LocalModel",
    "AdvancedLocalModel",
    "create_model",
    "FLClient",
    "ClientTrainer",
    "FederatedLearningRunner",
    "create_sample_datasets",
    "ConfigManager",
    "ModelConfig",
    "DataConfig", 
    "FLConfig",
    "SystemConfig"
]
