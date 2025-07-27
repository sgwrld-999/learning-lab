"""
Local Model Definition for Federated Learning

This module contains flexible PyTorch model architectures that can adapt
to different input dimensions and output classes for federated learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalModel(nn.Module):
    """
    Flexible neural network model for federated learning.
    
    This model can adapt to different input dimensions and output classes,
    making it suitable for heterogeneous federated learning scenarios.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Optional[list] = None, dropout_rate: float = 0.2):
        """
        Initialize the local model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions (default: [64, 32])
            dropout_rate: Dropout rate for regularization
        """
        super(LocalModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "total_params": total_params,
            "trainable_params": trainable_params
        }


class AdvancedLocalModel(nn.Module):
    """
    More advanced model with batch normalization and residual connections.
    Suitable for larger datasets and more complex federated learning scenarios.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Optional[list] = None, 
                 use_batch_norm: bool = True, use_residual: bool = False):
        """
        Initialize the advanced local model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super(AdvancedLocalModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the advanced model."""
        # Input layer
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        
        # Hidden layers with optional batch norm and residual connections
        for i, layer in enumerate(self.hidden_layers):
            residual = x if self.use_residual and x.shape[1] == layer.out_features else None
            
            x = layer(x)
            
            if self.use_batch_norm and self.batch_norms:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            
            if residual is not None:
                x = x + residual
            
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


def create_model(model_type: str = "simple", **kwargs) -> nn.Module:
    """
    Factory function to create different types of models.
    
    Args:
        model_type: Type of model to create ("simple" or "advanced")
        **kwargs: Model-specific arguments
        
    Returns:
        Instantiated model
    """
    if model_type.lower() == "simple":
        return LocalModel(**kwargs)
    elif model_type.lower() == "advanced":
        return AdvancedLocalModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
