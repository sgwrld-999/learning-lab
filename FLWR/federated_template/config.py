"""
Configuration and Utilities for Federated Learning Template

This module provides configuration management and utility functions
for the federated learning system.
"""

import os
import json
import logging
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    type: str = "simple"
    hidden_dims: list = None
    dropout_rate: float = 0.2
    use_batch_norm: bool = False
    use_residual: bool = False
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


@dataclass
class DataConfig:
    """Configuration for data handling."""
    batch_size: int = 32
    test_size: float = 0.2
    random_state: int = 42
    normalize: bool = True


@dataclass
class FLConfig:
    """Configuration for federated learning."""
    num_rounds: int = 3
    local_epochs: int = 1
    min_fit_clients: int = 2
    min_eval_clients: int = 2
    min_available_clients: int = 2
    strategy: str = "FedAvg"


@dataclass
class SystemConfig:
    """System-level configuration."""
    device: str = "auto"  # "auto", "cpu", "cuda"
    log_level: str = "INFO"
    seed: int = 42
    output_dir: str = "outputs"


class ConfigManager:
    """Manages configuration for the federated learning system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.fl_config = FLConfig()
        self.system_config = SystemConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if 'model' in config_dict:
            self.model_config = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            self.data_config = DataConfig(**config_dict['data'])
        if 'federated_learning' in config_dict:
            self.fl_config = FLConfig(**config_dict['federated_learning'])
        if 'system' in config_dict:
            self.system_config = SystemConfig(**config_dict['system'])
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'model': asdict(self.model_config),
            'data': asdict(self.data_config),
            'federated_learning': asdict(self.fl_config),
            'system': asdict(self.system_config)
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_device(self) -> torch.device:
        """Get PyTorch device based on configuration."""
        if self.system_config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.system_config.device)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_default_config(output_path: str = "config.json") -> None:
    """Create a default configuration file."""
    config_manager = ConfigManager()
    config_manager.save_to_file(output_path)
    print(f"Default configuration saved to {output_path}")


def validate_datasets(dataset_paths: list, target_col: str = "target") -> bool:
    """
    Validate that all datasets exist and have the required target column.
    
    Args:
        dataset_paths: List of dataset file paths
        target_col: Name of target column
        
    Returns:
        True if all datasets are valid
    """
    import pandas as pd
    
    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"Error: Dataset not found: {path}")
            return False
        
        try:
            df = pd.read_csv(path)
            if target_col not in df.columns:
                print(f"Error: Target column '{target_col}' not found in {path}")
                return False
            
            if len(df) < 10:
                print(f"Warning: Dataset {path} has very few samples ({len(df)})")
        
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return False
    
    return True


def print_system_info() -> None:
    """Print system information useful for debugging."""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    try:
        import flwr
        print(f"Flower version: {flwr.__version__}")
    except ImportError:
        print("Flower not installed")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    # Create default configuration file
    create_default_config()
    print_system_info()
