# Generic Federated Learning Template using FLWR + PyTorch

A modular, reusable federated learning system using Flower (FLWR) and PyTorch for handling heterogeneous datasets with different feature spaces but common target columns.

## Overview

This template is designed for real-world federated learning scenarios where:
- Multiple clients have datasets with different features (heterogeneous feature spaces)
- All datasets share a common target column for prediction
- Clients need to train collaboratively without sharing raw data
- The system should be scalable and adaptable to various domains (IoT, healthcare, finance, etc.)

## Key Features

- **Dataset-Agnostic**: Handles CSV datasets with varying feature sets
- **Automatic Feature Alignment**: Aligns heterogeneous feature spaces across clients
- **Flexible Model Architecture**: Supports simple and advanced neural network models
- **Comprehensive Logging**: Tracks training progress and evaluation metrics
- **Easy Configuration**: Command-line interface with sensible defaults
- **Simulation Ready**: Built-in support for Flower simulation framework

## Architecture

### Core Components

1. **DatasetManager** (`dataset_manager.py`)
   - Loads and aligns multiple CSV datasets
   - Handles feature space heterogeneity
   - Provides train/test splits with normalization

2. **LocalModel** (`model.py`)
   - Flexible PyTorch neural networks
   - Adapts to different input/output dimensions
   - Supports both simple and advanced architectures

3. **FLClient** (`client.py`)
   - Implements Flower NumPyClient interface
   - Handles local training and evaluation
   - Manages parameter synchronization

4. **FederatedLearningRunner** (`main.py`)
   - Orchestrates the entire FL process
   - Configures strategy and simulation
   - Provides comprehensive logging

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Create Sample Data (Optional)

```bash
python main.py --create-sample
```

This creates 3 sample datasets with heterogeneous features in a `sample_data` directory.

### 3. Run Federated Learning

```bash
# Using sample data
python main.py --create-sample --num-rounds 5

# Using your own datasets
python main.py --dataset-paths dataset1.csv dataset2.csv dataset3.csv --num-rounds 3

# Using all CSV files in a directory
python main.py --data-dir /path/to/csv/files --num-rounds 5
```

## Usage Examples

### Basic Usage

```python
from federated_template.main import FederatedLearningRunner

# Initialize with your datasets
dataset_paths = ["client1.csv", "client2.csv", "client3.csv"]
fl_runner = FederatedLearningRunner(dataset_paths, target_col="target")

# Run federated learning
fl_runner.run_simulation(num_rounds=5)
```

### Advanced Configuration

```python
# Custom model configuration
model_config = {
    "type": "advanced",
    "hidden_dims": [128, 64, 32],
    "use_batch_norm": True,
    "use_residual": False
}

# Custom data configuration
data_config = {
    "batch_size": 64,
    "test_size": 0.3
}

# Custom strategy configuration
strategy_config = {
    "local_epochs": 3,
    "min_fit_clients": 2,
    "min_eval_clients": 2
}

# Run with custom configurations
fl_runner.run_simulation(
    num_rounds=10,
    model_config=model_config,
    data_config=data_config,
    strategy_config=strategy_config
)
```

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --data-dir TEXT           Directory containing CSV datasets
  --dataset-paths TEXT...   Paths to specific CSV files
  --target-col TEXT         Name of target column [default: target]
  --num-rounds INTEGER      Number of FL rounds [default: 3]
  --create-sample          Create sample datasets
  --model-type [simple|advanced]  Model architecture type [default: simple]
  --batch-size INTEGER     Batch size for training [default: 32]
  --local-epochs INTEGER   Local epochs per round [default: 1]
```

## Dataset Requirements

Your CSV datasets should:
- Have a target column (default name: "target")
- Contain numeric features (categorical features should be encoded)
- Have consistent target values across all datasets
- Include at least 50+ samples for meaningful training

### Example Dataset Structure

**Client 1 Dataset (client1.csv):**
```csv
feature_A,feature_B,common_feature,target
1.2,3.4,0.5,0
2.1,1.8,0.7,1
...
```

**Client 2 Dataset (client2.csv):**
```csv
feature_C,feature_D,common_feature,special_feature,target
0.8,2.1,0.6,1.1,0
1.5,0.9,0.4,2.3,1
...
```

The system automatically aligns these heterogeneous feature spaces.

## Model Architecture

### Simple Model (Default)
- 2-3 hidden layers with ReLU activation
- Dropout for regularization
- Suitable for most datasets

### Advanced Model
- Deeper architecture with batch normalization
- Optional residual connections
- Better for larger, more complex datasets

## Extending the Template

### Custom Models

```python
from federated_template.model import LocalModel

class CustomModel(LocalModel):
    def __init__(self, input_dim, num_classes):
        super().__init__(input_dim, num_classes)
        # Add your custom layers
```

### Custom Strategies

```python
import flwr as fl

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Custom aggregation logic
        return super().aggregate_fit(server_round, results, failures)
```

## Output and Logging

The system provides comprehensive logging including:
- Dataset information and alignment details
- Client-wise training progress
- Round-by-round loss and accuracy metrics
- Final simulation results

## Real-World Applications

This template is suitable for:
- **Industrial IoT**: Sensor data from different manufacturing sites
- **Healthcare**: Medical data from different hospitals with varying features
- **Finance**: Transaction data from different banks or regions
- **Smart Cities**: Data from different urban sensors and systems

## Contributing

To extend this template:
1. Fork the repository
2. Add your improvements
3. Test with different datasets
4. Submit a pull request

## License

This template is provided as-is for educational and research purposes.

## Support

For issues or questions, please refer to:
- [Flower Documentation](https://flower.ai/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- This repository's issue tracker
