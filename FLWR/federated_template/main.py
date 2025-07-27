"""
Main Federated Learning Simulation Runner

This module orchestrates the entire federated learning process using Flower,
managing multiple clients with heterogeneous datasets and coordinating training.
"""

import os
import sys
import torch
import flwr as fl
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable
import argparse
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_manager import DatasetManager
from model import LocalModel, create_model
from client import FLClient


class FederatedLearningRunner:
    """
    Main class for running federated learning simulations.
    
    Handles the orchestration of multiple clients, server configuration,
    and the overall federated learning process.
    """
    
    def __init__(self, dataset_paths: List[str], target_col: str = "target",
                 device: Optional[torch.device] = None):
        """
        Initialize the federated learning runner.
        
        Args:
            dataset_paths: List of paths to CSV datasets
            target_col: Name of the target column
            device: Device to run computation on (CPU/GPU)
        """
        self.dataset_paths = dataset_paths
        self.target_col = target_col
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager(dataset_paths, target_col)
        self.dataset_manager.align_and_prepare()
        
        # Print dataset information
        self._print_dataset_info()
    
    def _print_dataset_info(self):
        """Print information about the loaded datasets."""
        info = self.dataset_manager.get_dataset_info()
        print("\n" + "="*60)
        print("FEDERATED LEARNING SETUP")
        print("="*60)
        print(f"Number of clients (datasets): {info['num_datasets']}")
        print(f"Total features after alignment: {info['num_features']}")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Device: {self.device}")
        print(f"Class labels: {info['class_labels']}")
        
        print("\nDataset shapes:")
        for i, shape in enumerate(info['dataset_shapes']):
            print(f"  Client {i+1}: {shape}")
        print("="*60 + "\n")
    
    def create_client_fn(self, model_config: Dict, data_config: Dict) -> Callable:
        """
        Create a client function for Flower simulation.
        
        Args:
            model_config: Configuration for model creation
            data_config: Configuration for data loading
            
        Returns:
            Client function for Flower simulation
        """
        def client_fn(cid: str) -> FLClient:
            """Create a client instance for the given client ID."""
            client_idx = int(cid)
            
            # Get data partition for this client
            trainset, testset = self.dataset_manager.get_partition(
                client_idx, 
                test_size=data_config.get("test_size", 0.2),
                random_state=data_config.get("random_state", 42)
            )
            
            # Create data loaders
            trainloader = DataLoader(
                trainset, 
                batch_size=data_config.get("batch_size", 32), 
                shuffle=True
            )
            testloader = DataLoader(
                testset, 
                batch_size=data_config.get("batch_size", 32), 
                shuffle=False
            )
            
            # Create model
            input_dim = self.dataset_manager.get_feature_count()
            num_classes = self.dataset_manager.get_class_count()
            
            model = create_model(
                model_type=model_config.get("type", "simple"),
                input_dim=input_dim,
                num_classes=num_classes,
                **{k: v for k, v in model_config.items() if k != "type"}
            )
            
            # Create and return client
            client = FLClient(
                model=model,
                trainloader=trainloader,
                testloader=testloader,
                device=self.device,
                client_id=f"client_{client_idx}"
            )
            
            print(f"Created client {client_idx} with {len(trainset)} train samples and {len(testset)} test samples")
            
            return client
        
        return client_fn
    
    def run_simulation(self, num_rounds: int = 3, 
                      model_config: Optional[Dict] = None,
                      data_config: Optional[Dict] = None,
                      strategy_config: Optional[Dict] = None) -> None:
        """
        Run the federated learning simulation.
        
        Args:
            num_rounds: Number of federated learning rounds
            model_config: Configuration for model creation
            data_config: Configuration for data loading
            strategy_config: Configuration for FL strategy
        """
        # Set default configurations
        if model_config is None:
            model_config = {"type": "simple", "hidden_dims": [64, 32]}
        
        if data_config is None:
            data_config = {"batch_size": 32, "test_size": 0.2}
        
        if strategy_config is None:
            strategy_config = {
                "min_fit_clients": len(self.dataset_paths),
                "min_eval_clients": len(self.dataset_paths),
                "min_available_clients": len(self.dataset_paths),
                "local_epochs": 1
            }
        
        print(f"Starting federated learning simulation with {num_rounds} rounds...")
        print(f"Model config: {model_config}")
        print(f"Data config: {data_config}")
        print(f"Strategy config: {strategy_config}")
        
        # Create client function
        client_fn = self.create_client_fn(model_config, data_config)
        
        # Define strategy configuration function
        def fit_config(server_round: int) -> Dict:
            """Return training configuration dict for each round."""
            config = {
                "local_epochs": strategy_config.get("local_epochs", 1),
                "verbose": server_round <= 2  # Verbose for first 2 rounds
            }
            return config
        
        def evaluate_config(server_round: int) -> Dict:
            """Return evaluation configuration dict for each round."""
            return {"verbose": True}
        
        # Create strategy
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=strategy_config["min_fit_clients"],
            min_eval_clients=strategy_config["min_eval_clients"],
            min_available_clients=strategy_config["min_available_clients"],
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=None,
        )
        
        # Run simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(self.dataset_paths),
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)
        self._print_results(history)
    
    def _print_results(self, history) -> None:
        """Print simulation results."""
        print(f"Total rounds completed: {len(history.losses_distributed)}")
        
        if history.losses_distributed:
            print("\nTraining Loss History:")
            for round_num, (loss, _) in enumerate(history.losses_distributed):
                print(f"  Round {round_num + 1}: {loss:.4f}")
        
        if history.metrics_distributed:
            print("\nAccuracy History:")
            for round_num, metrics in history.metrics_distributed.items():
                if 'accuracy' in metrics:
                    acc = metrics['accuracy'][0] if isinstance(metrics['accuracy'], tuple) else metrics['accuracy']
                    print(f"  Round {round_num + 1}: {acc:.4f}")


def create_sample_datasets(output_dir: str = "sample_data", num_datasets: int = 3) -> List[str]:
    """
    Create sample datasets for testing the federated learning system.
    
    Args:
        output_dir: Directory to save sample datasets
        num_datasets: Number of datasets to create
        
    Returns:
        List of paths to created datasets
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    
    os.makedirs(output_dir, exist_ok=True)
    dataset_paths = []
    
    # Base features that all datasets share
    base_features = ['feature_1', 'feature_2', 'feature_3']
    
    for i in range(num_datasets):
        # Generate synthetic data
        n_samples = np.random.randint(200, 500)
        n_features = np.random.randint(5, 10)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features, 5),
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=3,
            random_state=42 + i
        )
        
        # Create feature names (some unique, some shared)
        unique_features = [f'unique_feature_{i}_{j}' for j in range(n_features - len(base_features))]
        all_features = base_features + unique_features
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=all_features)
        df['target'] = y
        
        # Save dataset
        filepath = os.path.join(output_dir, f'dataset_{i+1}.csv')
        df.to_csv(filepath, index=False)
        dataset_paths.append(filepath)
        
        print(f"Created dataset {i+1}: {filepath} with shape {df.shape}")
    
    return dataset_paths


def main():
    """Main function to run federated learning simulation."""
    parser = argparse.ArgumentParser(description='Run Federated Learning Simulation')
    parser.add_argument('--data-dir', type=str, help='Directory containing CSV datasets')
    parser.add_argument('--dataset-paths', nargs='+', help='Paths to specific CSV files')
    parser.add_argument('--target-col', type=str, default='target', help='Name of target column')
    parser.add_argument('--num-rounds', type=int, default=3, help='Number of FL rounds')
    parser.add_argument('--create-sample', action='store_true', help='Create sample datasets')
    parser.add_argument('--model-type', type=str, default='simple', choices=['simple', 'advanced'])
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--local-epochs', type=int, default=1, help='Local epochs per round')
    
    args = parser.parse_args()
    
    # Determine dataset paths
    if args.create_sample:
        print("Creating sample datasets...")
        dataset_paths = create_sample_datasets()
    elif args.dataset_paths:
        dataset_paths = args.dataset_paths
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        dataset_paths = [str(p) for p in data_dir.glob('*.csv')]
        if not dataset_paths:
            raise ValueError(f"No CSV files found in {args.data_dir}")
    else:
        raise ValueError("Must specify either --dataset-paths, --data-dir, or --create-sample")
    
    print(f"Using datasets: {dataset_paths}")
    
    # Create federated learning runner
    fl_runner = FederatedLearningRunner(dataset_paths, args.target_col)
    
    # Configuration
    model_config = {"type": args.model_type}
    data_config = {"batch_size": args.batch_size}
    strategy_config = {"local_epochs": args.local_epochs}
    
    # Run simulation
    fl_runner.run_simulation(
        num_rounds=args.num_rounds,
        model_config=model_config,
        data_config=data_config,
        strategy_config=strategy_config
    )


if __name__ == "__main__":
    main()
