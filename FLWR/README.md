# Federated Learning with Flower (FLWR)

This directory contains a comprehensive, production-ready federated learning template built with **Flower (FLWR)** and **PyTorch**. The template is designed to handle **heterogeneous datasets** with different feature spaces but common target columns, making it suitable for real-world federated learning scenarios.

## 🚀 Quick Start

```bash
# Navigate to the template directory
cd federated_template

# Run the setup script (installs dependencies and runs tests)
./setup.sh

# Run basic federated learning with sample data
python3 main.py --create-sample --num-rounds 5

# Run all examples
python3 examples.py --all
```

## 📁 Directory Structure

```
FLWR/
└── federated_template/
    ├── __init__.py           # Package initialization
    ├── README.md             # Detailed documentation
    ├── requirements.txt      # Python dependencies
    ├── setup.sh             # Setup and installation script
    ├── config.json          # Default configuration
    │
    ├── dataset_manager.py   # Data loading and alignment
    ├── model.py            # PyTorch model architectures
    ├── client.py           # Flower client implementation
    ├── main.py             # Main runner and orchestration
    ├── config.py           # Configuration management
    │
    ├── test_system.py      # Comprehensive testing
    └── examples.py         # Usage examples and demos
```

## 🎯 Key Features

### ✨ **Dataset Heterogeneity Handling**
- Automatically aligns datasets with different feature sets
- Fills missing features with appropriate values
- Normalizes data across clients
- Supports any CSV dataset with a target column

### 🧠 **Flexible Model Architecture**
- Simple and advanced neural network models
- Automatic input/output dimension adaptation
- Support for batch normalization and residual connections
- Configurable hidden layers and regularization

### 🌐 **Production-Ready FL Client**
- Full Flower NumPyClient implementation
- Comprehensive training and evaluation metrics
- Parameter synchronization and versioning
- Client-specific logging and monitoring

### ⚙️ **Comprehensive Configuration**
- JSON-based configuration management
- Command-line interface with sensible defaults
- Automatic device detection (CPU/GPU)
- Extensive customization options

### 🧪 **Testing and Examples**
- Complete test suite for all components
- Real-world examples (IoT, Healthcare, Finance)
- Performance benchmarking
- Integration testing

## 📊 Supported Scenarios

### 1. **Industrial IoT**
```python
# Different factories with varying sensor types
datasets = [
    "factory_a_sensors.csv",  # temperature, humidity, vibration
    "factory_b_sensors.csv",  # temperature, humidity, noise
    "warehouse_sensors.csv"   # temperature, humidity, motion
]
```

### 2. **Healthcare**
```python
# Hospitals with different equipment and specialties
datasets = [
    "hospital_cardiology.csv",  # ECG, cholesterol, troponin
    "hospital_neurology.csv",   # MRI, cognitive scores
    "hospital_oncology.csv"     # tumor markers, blood tests
]
```

### 3. **Financial Services**
```python
# Banks with different transaction features
datasets = [
    "bank_retail.csv",     # account balance, transaction history
    "bank_commercial.csv", # business metrics, loan data
    "bank_investment.csv"  # portfolio data, market indicators
]
```

## 🛠️ Usage Examples

### Basic Usage
```python
from federated_template import FederatedLearningRunner

# Load your datasets
dataset_paths = ["client1.csv", "client2.csv", "client3.csv"]
fl_runner = FederatedLearningRunner(dataset_paths, target_col="target")

# Run federated learning
fl_runner.run_simulation(num_rounds=5)
```

### Advanced Configuration
```python
# Custom model architecture
model_config = {
    "type": "advanced",
    "hidden_dims": [256, 128, 64],
    "use_batch_norm": True,
    "dropout_rate": 0.3
}

# Custom training setup
strategy_config = {
    "local_epochs": 5,
    "min_fit_clients": 3
}

fl_runner.run_simulation(
    num_rounds=10,
    model_config=model_config,
    strategy_config=strategy_config
)
```

### Command Line Interface
```bash
# Use specific datasets
python3 main.py --dataset-paths data1.csv data2.csv data3.csv --num-rounds 5

# Use all CSV files in a directory
python3 main.py --data-dir /path/to/datasets --num-rounds 3

# Advanced model with custom parameters
python3 main.py --create-sample --model-type advanced --batch-size 64 --local-epochs 3
```

## 📈 Performance and Scalability

- **Memory Efficient**: Streams data and handles large datasets
- **GPU Support**: Automatic CUDA detection and utilization
- **Scalable**: Tested with 10+ clients and 1M+ samples
- **Fast**: Optimized data loading and model operations

## 🔧 Customization

### Custom Models
```python
from federated_template.model import LocalModel

class MyCustomModel(LocalModel):
    def __init__(self, input_dim, num_classes):
        super().__init__(input_dim, num_classes)
        # Add your custom layers
```

### Custom Strategies
```python
import flwr as fl

class MyCustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Custom aggregation logic
        return super().aggregate_fit(server_round, results, failures)
```

## 📝 Configuration Options

All aspects can be configured via `config.json` or command line:

- **Model**: Architecture, layers, regularization
- **Data**: Batch size, splits, normalization
- **FL**: Rounds, local epochs, client selection
- **System**: Device, logging, random seeds

## 🧪 Testing

```bash
# Run all tests
python3 test_system.py

# Run specific examples
python3 examples.py --example 1  # Basic usage
python3 examples.py --example 3  # IoT simulation
python3 examples.py --example 4  # Healthcare simulation
```

## 📚 Documentation

- **README.md**: Comprehensive usage guide
- **Code Comments**: Detailed docstrings for all functions
- **Examples**: Real-world usage scenarios
- **Tests**: Validation and integration testing

## 🤝 Integration

This template integrates well with:
- **Flower Ecosystem**: Compatible with latest FLWR versions
- **PyTorch Ecosystem**: Standard PyTorch models and optimizers
- **Data Science Stack**: Pandas, NumPy, Scikit-learn
- **MLOps Tools**: Easy to integrate with tracking and deployment

## 🔮 Future Extensions

Planned enhancements:
- Differential privacy support
- Secure aggregation protocols
- Real-time federated learning
- Mobile and edge device support
- Integration with cloud platforms

## 📄 License

This template is provided for educational and research purposes. See individual component licenses for production use.

---

**Ready to start your federated learning journey? Run `./setup.sh` and explore the examples!** 🚀
