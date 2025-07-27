"""
Test script for the Federated Learning Template

This script provides comprehensive testing for all components
of the federated learning system.
"""

import os
import sys
import tempfile
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_manager import DatasetManager
from model import LocalModel, AdvancedLocalModel, create_model
from client import FLClient, ClientTrainer
from main import FederatedLearningRunner, create_sample_datasets
from config import ConfigManager, ModelConfig, DataConfig, FLConfig


class TestDatasetManager(unittest.TestCase):
    """Test cases for DatasetManager."""
    
    def setUp(self):
        """Set up test datasets."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_paths = self._create_test_datasets()
    
    def _create_test_datasets(self):
        """Create temporary test datasets."""
        from sklearn.datasets import make_classification
        
        paths = []
        base_features = ['feat_A', 'feat_B']
        
        for i in range(3):
            # Generate data
            X, y = make_classification(
                n_samples=100,
                n_features=5,
                n_informative=3,
                n_classes=2,
                random_state=42 + i
            )
            
            # Create unique feature names
            unique_features = [f'unique_{i}_{j}' for j in range(3)]
            all_features = base_features + unique_features
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=all_features)
            df['target'] = y
            
            # Save dataset
            path = os.path.join(self.temp_dir, f'test_dataset_{i}.csv')
            df.to_csv(path, index=False)
            paths.append(path)
        
        return paths
    
    def test_initialization(self):
        """Test DatasetManager initialization."""
        dm = DatasetManager(self.dataset_paths)
        self.assertEqual(len(dm.datasets), 3)
        self.assertEqual(dm.target_col, "target")
    
    def test_align_and_prepare(self):
        """Test feature alignment functionality."""
        dm = DatasetManager(self.dataset_paths)
        dm.align_and_prepare()
        
        # Check that all datasets have the same number of features
        feature_counts = [len(df.columns) - 1 for df in dm.aligned]  # -1 for target
        self.assertTrue(all(count == feature_counts[0] for count in feature_counts))
        
        # Check that all datasets have the same feature names (except target)
        for df in dm.aligned:
            feature_names = [col for col in df.columns if col != 'target']
            self.assertEqual(feature_names, dm.all_features)
    
    def test_get_partition(self):
        """Test data partitioning for clients."""
        dm = DatasetManager(self.dataset_paths)
        dm.align_and_prepare()
        
        train_dataset, test_dataset = dm.get_partition(0)
        
        # Check dataset types
        self.assertEqual(train_dataset.__class__.__name__, 'CustomDataset')
        self.assertEqual(test_dataset.__class__.__name__, 'CustomDataset')
        
        # Check that we have both train and test data
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestModels(unittest.TestCase):
    """Test cases for model architectures."""
    
    def test_local_model_creation(self):
        """Test LocalModel creation and basic functionality."""
        model = LocalModel(input_dim=10, num_classes=3)
        
        self.assertEqual(model.input_dim, 10)
        self.assertEqual(model.num_classes, 3)
        
        # Test forward pass
        import torch
        x = torch.randn(5, 10)
        output = model(x)
        self.assertEqual(output.shape, (5, 3))
    
    def test_advanced_model_creation(self):
        """Test AdvancedLocalModel creation."""
        model = AdvancedLocalModel(
            input_dim=15, 
            num_classes=4, 
            hidden_dims=[32, 16],
            use_batch_norm=True
        )
        
        # Test forward pass
        import torch
        x = torch.randn(8, 15)
        output = model(x)
        self.assertEqual(output.shape, (8, 4))
    
    def test_model_factory(self):
        """Test model factory function."""
        simple_model = create_model("simple", input_dim=5, num_classes=2)
        advanced_model = create_model("advanced", input_dim=5, num_classes=2)
        
        self.assertIsInstance(simple_model, LocalModel)
        self.assertIsInstance(advanced_model, AdvancedLocalModel)


class TestClientTrainer(unittest.TestCase):
    """Test cases for ClientTrainer."""
    
    def setUp(self):
        """Set up test components."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        self.device = torch.device("cpu")
        self.model = LocalModel(input_dim=5, num_classes=2)
        
        # Create dummy data
        X = torch.randn(50, 5)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=10)
        
        self.trainer = ClientTrainer(self.model, self.device)
    
    def test_trainer_initialization(self):
        """Test ClientTrainer initialization."""
        self.assertEqual(self.trainer.device, self.device)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        metrics = self.trainer.train_epoch(self.dataloader)
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_evaluate(self):
        """Test model evaluation."""
        metrics = self.trainer.evaluate(self.dataloader)
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)


class TestConfig(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_model_config(self):
        """Test ModelConfig dataclass."""
        config = ModelConfig()
        self.assertEqual(config.type, "simple")
        self.assertEqual(config.hidden_dims, [64, 32])
    
    def test_config_manager(self):
        """Test ConfigManager functionality."""
        config_manager = ConfigManager()
        
        # Test that all configs are initialized
        self.assertIsNotNone(config_manager.model_config)
        self.assertIsNotNone(config_manager.data_config)
        self.assertIsNotNone(config_manager.fl_config)
        self.assertIsNotNone(config_manager.system_config)
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            config_manager = ConfigManager()
            config_manager.save_to_file(config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Load config
            new_config_manager = ConfigManager(config_path)
            self.assertEqual(
                config_manager.model_config.type,
                new_config_manager.model_config.type
            )
        finally:
            os.unlink(config_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = create_sample_datasets(temp_dir, num_datasets=2)
            
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(os.path.exists(path))
                df = pd.read_csv(path)
                self.assertIn('target', df.columns)
                self.assertGreater(len(df), 0)
    
    def test_federated_learning_runner_initialization(self):
        """Test FederatedLearningRunner initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = create_sample_datasets(temp_dir, num_datasets=2)
            
            fl_runner = FederatedLearningRunner(paths)
            
            self.assertEqual(len(fl_runner.dataset_paths), 2)
            self.assertIsNotNone(fl_runner.dataset_manager)
            self.assertGreater(fl_runner.dataset_manager.get_feature_count(), 0)
            self.assertGreater(fl_runner.dataset_manager.get_class_count(), 0)


def run_basic_simulation_test():
    """
    Run a basic simulation test to ensure everything works together.
    This is separate from unittest to avoid long test times.
    """
    print("Running basic simulation test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample datasets
        paths = create_sample_datasets(temp_dir, num_datasets=2)
        
        # Initialize runner
        fl_runner = FederatedLearningRunner(paths)
        
        # Run short simulation
        fl_runner.run_simulation(
            num_rounds=1,
            model_config={"type": "simple", "hidden_dims": [16]},
            data_config={"batch_size": 16},
            strategy_config={"local_epochs": 1}
        )
        
        print("Basic simulation test completed successfully!")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "="*60)
    run_basic_simulation_test()
