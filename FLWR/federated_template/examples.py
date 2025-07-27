"""
Example Usage Scripts for Federated Learning Template

This module contains practical examples showing how to use
the federated learning template in different scenarios.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add template to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FederatedLearningRunner
from config import ConfigManager


def example_1_basic_usage():
    """
    Example 1: Basic federated learning with sample datasets
    """
    print("="*60)
    print("EXAMPLE 1: Basic Federated Learning")
    print("="*60)
    
    # Create sample datasets
    from main import create_sample_datasets
    dataset_paths = create_sample_datasets("example_data", num_datasets=3)
    
    # Initialize federated learning
    fl_runner = FederatedLearningRunner(dataset_paths, target_col="target")
    
    # Run basic simulation
    fl_runner.run_simulation(num_rounds=3)
    
    print("\n✅ Example 1 completed successfully!\n")


def example_2_custom_configuration():
    """
    Example 2: Custom model and training configuration
    """
    print("="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Create datasets
    from main import create_sample_datasets
    dataset_paths = create_sample_datasets("example_data_2", num_datasets=4)
    
    # Initialize with custom configuration
    fl_runner = FederatedLearningRunner(dataset_paths)
    
    # Custom configurations
    model_config = {
        "type": "advanced",
        "hidden_dims": [128, 64, 32],
        "use_batch_norm": True,
        "dropout_rate": 0.3
    }
    
    data_config = {
        "batch_size": 64,
        "test_size": 0.25
    }
    
    strategy_config = {
        "local_epochs": 3,
        "min_fit_clients": 3,
        "min_eval_clients": 3
    }
    
    # Run simulation with custom config
    fl_runner.run_simulation(
        num_rounds=5,
        model_config=model_config,
        data_config=data_config,
        strategy_config=strategy_config
    )
    
    print("\n✅ Example 2 completed successfully!\n")


def example_3_iot_sensor_simulation():
    """
    Example 3: Simulate IoT sensor data from different locations
    """
    print("="*60)
    print("EXAMPLE 3: IoT Sensor Simulation")
    print("="*60)
    
    # Create IoT-like datasets
    datasets = create_iot_sensor_datasets()
    
    # Initialize federated learning
    fl_runner = FederatedLearningRunner(datasets, target_col="anomaly")
    
    # IoT-specific configuration
    model_config = {
        "type": "simple",
        "hidden_dims": [32, 16],  # Smaller model for edge devices
        "dropout_rate": 0.1
    }
    
    data_config = {
        "batch_size": 16,  # Smaller batches for resource-constrained devices
        "test_size": 0.2
    }
    
    strategy_config = {
        "local_epochs": 2,
        "min_fit_clients": 3
    }
    
    # Run simulation
    fl_runner.run_simulation(
        num_rounds=4,
        model_config=model_config,
        data_config=data_config,
        strategy_config=strategy_config
    )
    
    print("\n✅ Example 3 completed successfully!\n")


def example_4_healthcare_simulation():
    """
    Example 4: Simulate healthcare data from different hospitals
    """
    print("="*60)
    print("EXAMPLE 4: Healthcare Data Simulation")
    print("="*60)
    
    # Create healthcare-like datasets
    datasets = create_healthcare_datasets()
    
    # Initialize federated learning
    fl_runner = FederatedLearningRunner(datasets, target_col="diagnosis")
    
    # Healthcare-specific configuration (more complex model)
    model_config = {
        "type": "advanced",
        "hidden_dims": [256, 128, 64, 32],
        "use_batch_norm": True,
        "dropout_rate": 0.4  # Higher dropout for privacy
    }
    
    data_config = {
        "batch_size": 32,
        "test_size": 0.3
    }
    
    strategy_config = {
        "local_epochs": 5,  # More local training for privacy
        "min_fit_clients": 2
    }
    
    # Run simulation
    fl_runner.run_simulation(
        num_rounds=6,
        model_config=model_config,
        data_config=data_config,
        strategy_config=strategy_config
    )
    
    print("\n✅ Example 4 completed successfully!\n")


def create_iot_sensor_datasets():
    """Create synthetic IoT sensor datasets."""
    output_dir = "iot_sensor_data"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = []
    
    # Different sensor types at different locations
    sensor_configs = [
        {"location": "factory_a", "sensors": ["temperature", "humidity", "vibration", "pressure"]},
        {"location": "factory_b", "sensors": ["temperature", "humidity", "noise", "light"]},
        {"location": "warehouse", "sensors": ["temperature", "humidity", "motion", "air_quality"]},
        {"location": "office", "sensors": ["temperature", "humidity", "co2", "occupancy"]}
    ]
    
    for i, config in enumerate(sensor_configs):
        # Generate sensor data
        n_samples = np.random.randint(300, 600)
        data = {}
        
        # Common sensors (temperature, humidity)
        data["temperature"] = np.random.normal(22, 5, n_samples)  # Celsius
        data["humidity"] = np.random.normal(45, 15, n_samples)    # Percentage
        
        # Location-specific sensors
        for sensor in config["sensors"][2:]:  # Skip temperature and humidity
            if sensor == "vibration":
                data[sensor] = np.random.exponential(2, n_samples)
            elif sensor == "pressure":
                data[sensor] = np.random.normal(1013, 20, n_samples)  # hPa
            elif sensor == "noise":
                data[sensor] = np.random.normal(40, 10, n_samples)  # dB
            elif sensor == "light":
                data[sensor] = np.random.normal(300, 100, n_samples)  # lux
            elif sensor == "motion":
                data[sensor] = np.random.binomial(1, 0.3, n_samples)
            elif sensor == "air_quality":
                data[sensor] = np.random.normal(50, 20, n_samples)  # AQI
            elif sensor == "co2":
                data[sensor] = np.random.normal(400, 100, n_samples)  # ppm
            elif sensor == "occupancy":
                data[sensor] = np.random.poisson(5, n_samples)
        
        # Create anomaly target (based on extreme values)
        anomaly_conditions = [
            data["temperature"] > 35,  # Too hot
            data["temperature"] < 5,   # Too cold
            data["humidity"] > 80,     # Too humid
            data["humidity"] < 10      # Too dry
        ]
        
        if "vibration" in data:
            anomaly_conditions.append(data["vibration"] > 8)
        if "pressure" in data:
            anomaly_conditions.append(np.abs(data["pressure"] - 1013) > 50)
        
        anomaly = np.zeros(n_samples)
        for condition in anomaly_conditions:
            anomaly = anomaly | condition
        
        data["anomaly"] = anomaly.astype(int)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filepath = os.path.join(output_dir, f"sensor_{config['location']}.csv")
        df.to_csv(filepath, index=False)
        datasets.append(filepath)
        
        print(f"Created IoT dataset: {filepath} with {len(df)} samples")
    
    return datasets


def create_healthcare_datasets():
    """Create synthetic healthcare datasets."""
    output_dir = "healthcare_data"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = []
    
    # Different hospitals with different equipment/tests
    hospital_configs = [
        {"name": "hospital_a", "specialties": ["cardiology", "general"]},
        {"name": "hospital_b", "specialties": ["neurology", "general"]},
        {"name": "hospital_c", "specialties": ["oncology", "general"]},
    ]
    
    for i, config in enumerate(hospital_configs):
        # Generate patient data
        n_samples = np.random.randint(400, 800)
        data = {}
        
        # Common features (all hospitals have these)
        data["age"] = np.random.normal(45, 20, n_samples).clip(18, 90)
        data["gender"] = np.random.binomial(1, 0.5, n_samples)  # 0=female, 1=male
        data["bmi"] = np.random.normal(25, 5, n_samples).clip(15, 50)
        data["blood_pressure_systolic"] = np.random.normal(120, 20, n_samples)
        data["blood_pressure_diastolic"] = np.random.normal(80, 15, n_samples)
        data["heart_rate"] = np.random.normal(70, 15, n_samples)
        
        # Specialty-specific features
        if "cardiology" in config["specialties"]:
            data["ecg_abnormal"] = np.random.binomial(1, 0.3, n_samples)
            data["cholesterol"] = np.random.normal(200, 40, n_samples)
            data["troponin"] = np.random.exponential(0.1, n_samples)
        
        if "neurology" in config["specialties"]:
            data["mri_lesions"] = np.random.poisson(2, n_samples)
            data["cognitive_score"] = np.random.normal(28, 3, n_samples).clip(0, 30)
            data["reflexes_abnormal"] = np.random.binomial(1, 0.2, n_samples)
        
        if "oncology" in config["specialties"]:
            data["tumor_markers"] = np.random.exponential(5, n_samples)
            data["white_blood_cells"] = np.random.normal(7000, 2000, n_samples)
            data["hemoglobin"] = np.random.normal(13, 2, n_samples)
        
        # Create diagnosis target (simplified: 0=healthy, 1=condition, 2=severe)
        diagnosis_prob = np.zeros(n_samples)
        
        # Risk factors increase probability
        diagnosis_prob += (data["age"] > 60) * 0.3
        diagnosis_prob += (data["bmi"] > 30) * 0.2
        diagnosis_prob += (data["blood_pressure_systolic"] > 140) * 0.2
        
        # Specialty-specific risk factors
        if "cardiology" in config["specialties"]:
            diagnosis_prob += data.get("ecg_abnormal", 0) * 0.4
            diagnosis_prob += (data.get("cholesterol", 0) > 240) * 0.3
        
        if "neurology" in config["specialties"]:
            diagnosis_prob += (data.get("mri_lesions", 0) > 3) * 0.5
            diagnosis_prob += data.get("reflexes_abnormal", 0) * 0.3
        
        if "oncology" in config["specialties"]:
            diagnosis_prob += (data.get("tumor_markers", 0) > 10) * 0.6
            diagnosis_prob += (data.get("white_blood_cells", 0) > 10000) * 0.2
        
        # Convert probabilities to categories
        diagnosis = np.zeros(n_samples)
        diagnosis[diagnosis_prob > 0.3] = 1
        diagnosis[diagnosis_prob > 0.6] = 2
        
        data["diagnosis"] = diagnosis.astype(int)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        filepath = os.path.join(output_dir, f"{config['name']}.csv")
        df.to_csv(filepath, index=False)
        datasets.append(filepath)
        
        print(f"Created healthcare dataset: {filepath} with {len(df)} samples")
    
    return datasets


def run_all_examples():
    """Run all examples in sequence."""
    print("🚀 Running all federated learning examples...\n")
    
    try:
        example_1_basic_usage()
        example_2_custom_configuration()
        example_3_iot_sensor_simulation()
        example_4_healthcare_simulation()
        
        print("🎉 All examples completed successfully!")
        print("\nYou can find the generated datasets in:")
        print("  - example_data/")
        print("  - example_data_2/")
        print("  - iot_sensor_data/")
        print("  - healthcare_data/")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run federated learning examples")
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4], 
                       help="Run specific example (1-4)")
    parser.add_argument("--all", action="store_true", 
                       help="Run all examples")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_examples()
    elif args.example == 1:
        example_1_basic_usage()
    elif args.example == 2:
        example_2_custom_configuration()
    elif args.example == 3:
        example_3_iot_sensor_simulation()
    elif args.example == 4:
        example_4_healthcare_simulation()
    else:
        print("Please specify --all or --example [1-4]")
        print("\nAvailable examples:")
        print("  1: Basic federated learning")
        print("  2: Custom configuration")
        print("  3: IoT sensor simulation")
        print("  4: Healthcare simulation")
