"""
Dataset Manager for Federated Learning with Heterogeneous Features

This module handles loading, aligning, and preparing multiple CSV datasets
with different feature spaces but common target columns for federated learning.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom PyTorch Dataset for handling feature-target pairs."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DatasetManager:
    """
    Manages multiple CSV datasets with heterogeneous features for federated learning.
    
    Handles:
    - Loading datasets from CSV files
    - Aligning feature spaces across datasets
    - Normalizing and preprocessing data
    - Creating train/test splits for each client
    """
    
    def __init__(self, csv_paths: List[str], target_col: str = "target"):
        """
        Initialize the DatasetManager.
        
        Args:
            csv_paths: List of paths to CSV files
            target_col: Name of the target column
        """
        self.csv_paths = csv_paths
        self.target_col = target_col
        self.datasets = []
        self.all_features = []
        self.aligned = []
        self.label_encoder = LabelEncoder()
        self.num_classes = 0
        
        # Load all datasets
        self._load_datasets()
    
    def _load_datasets(self) -> None:
        """Load all CSV datasets and handle basic validation."""
        for path in self.csv_paths:
            try:
                df = pd.read_csv(path)
                if self.target_col not in df.columns:
                    raise ValueError(f"Target column '{self.target_col}' not found in {path}")
                self.datasets.append(df)
                print(f"Loaded dataset: {path} with shape {df.shape}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
                raise
    
    def align_and_prepare(self) -> None:
        """
        Align feature spaces across all datasets and prepare for training.
        
        This method:
        1. Identifies all unique features across datasets
        2. Fills missing features with zeros
        3. Aligns column order
        4. Encodes target labels
        """
        print("Aligning feature spaces across datasets...")
        
        # Collect all unique features
        all_features = set()
        all_targets = []
        
        for df in self.datasets:
            features = set(df.columns) - {self.target_col}
            all_features.update(features)
            all_targets.extend(df[self.target_col].unique())
        
        self.all_features = sorted(list(all_features))
        print(f"Total unique features: {len(self.all_features)}")
        
        # Fit label encoder on all targets
        unique_targets = list(set(all_targets))
        self.label_encoder.fit(unique_targets)
        self.num_classes = len(unique_targets)
        print(f"Number of classes: {self.num_classes}")
        
        # Align datasets
        self.aligned = []
        for i, df in enumerate(self.datasets):
            # Add missing features with zeros
            aligned_df = df.copy()
            for feature in self.all_features:
                if feature not in aligned_df.columns:
                    aligned_df[feature] = 0.0
            
            # Reorder columns: features first, then target
            aligned_df = aligned_df[self.all_features + [self.target_col]]
            
            # Encode target labels
            aligned_df[self.target_col] = self.label_encoder.transform(aligned_df[self.target_col])
            
            self.aligned.append(aligned_df)
            print(f"Dataset {i+1} aligned: {aligned_df.shape}")
    
    def get_partition(self, idx: int, test_size: float = 0.2, random_state: int = 42) -> Tuple[CustomDataset, CustomDataset]:
        """
        Get train and test datasets for a specific client.
        
        Args:
            idx: Client index (0-based)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if idx >= len(self.aligned):
            raise IndexError(f"Client index {idx} out of range. Only {len(self.aligned)} datasets available.")
        
        df = self.aligned[idx].copy()
        
        # Split into features and targets
        X = df.drop(columns=[self.target_col]).values
        y = df[self.target_col].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create PyTorch datasets
        train_dataset = CustomDataset(X_train_scaled, y_train)
        test_dataset = CustomDataset(X_test_scaled, y_test)
        
        print(f"Client {idx+1} - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def get_feature_count(self) -> int:
        """Get the total number of features after alignment."""
        return len(self.all_features)
    
    def get_class_count(self) -> int:
        """Get the number of target classes."""
        return self.num_classes
    
    def get_dataset_info(self) -> dict:
        """Get summary information about all datasets."""
        info = {
            "num_datasets": len(self.datasets),
            "num_features": len(self.all_features),
            "num_classes": self.num_classes,
            "feature_names": self.all_features,
            "class_labels": list(self.label_encoder.classes_),
            "dataset_shapes": [df.shape for df in self.datasets]
        }
        return info
