"""
Federated Learning Client Implementation

This module implements the Flower client for federated learning,
handling local training, evaluation, and parameter synchronization.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import flwr as fl
import numpy as np
from collections import OrderedDict


class ClientTrainer:
    """
    Handles local training and evaluation for a federated learning client.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """
        Initialize the client trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to run training on (CPU/GPU)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.9
        )
    
    def train_epoch(self, trainloader: DataLoader, verbose: bool = False) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            trainloader: DataLoader for training data
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if verbose and batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.6f}')
        
        # Update learning rate
        self.scheduler.step()
        
        avg_loss = total_loss / len(trainloader)
        accuracy = correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self, testloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            testloader: DataLoader for test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = F.cross_entropy(output, target, reduction='sum')
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }


class FLClient(fl.client.NumPyClient):
    """
    Flower federated learning client implementation.
    
    This client handles parameter synchronization, local training,
    and evaluation in the federated learning process.
    """
    
    def __init__(self, model: torch.nn.Module, trainloader: DataLoader, 
                 testloader: DataLoader, device: torch.device, client_id: str):
        """
        Initialize the federated learning client.
        
        Args:
            model: PyTorch model for training
            trainloader: DataLoader for training data
            testloader: DataLoader for test data
            device: Device to run computation on
            client_id: Unique identifier for this client
        """
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.client_id = client_id
        
        # Initialize trainer
        self.trainer = ClientTrainer(model, device)
        
        # Track client metrics
        self.training_history = []
        self.evaluation_history = []
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of model parameters as numpy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays.
        
        Args:
            parameters: List of parameters as numpy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model with given parameters and configuration.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        print(f"Client {self.client_id}: Starting local training")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Get training configuration
        local_epochs = config.get("local_epochs", 1)
        verbose = config.get("verbose", False)
        
        # Perform local training
        for epoch in range(local_epochs):
            train_metrics = self.trainer.train_epoch(self.trainloader, verbose)
            self.training_history.append(train_metrics)
            
            if verbose:
                print(f"Client {self.client_id}, Epoch {epoch+1}/{local_epochs}: "
                      f"Loss: {train_metrics['loss']:.4f}, "
                      f"Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Return updated parameters and metrics
        num_examples = len(self.trainloader.dataset)
        final_metrics = self.training_history[-1] if self.training_history else {}
        
        print(f"Client {self.client_id}: Completed local training")
        
        return self.get_parameters(config), num_examples, final_metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model with given parameters.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        print(f"Client {self.client_id}: Starting evaluation")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Perform evaluation
        eval_metrics = self.trainer.evaluate(self.testloader)
        self.evaluation_history.append(eval_metrics)
        
        num_examples = len(self.testloader.dataset)
        
        print(f"Client {self.client_id}: Evaluation complete - "
              f"Loss: {eval_metrics['loss']:.4f}, "
              f"Accuracy: {eval_metrics['accuracy']:.4f}")
        
        return eval_metrics["loss"], num_examples, eval_metrics
    
    def get_client_info(self) -> Dict:
        """Get information about this client."""
        return {
            "client_id": self.client_id,
            "train_samples": len(self.trainloader.dataset),
            "test_samples": len(self.testloader.dataset),
            "device": str(self.device),
            "model_info": self.model.get_model_info() if hasattr(self.model, 'get_model_info') else None
        }
