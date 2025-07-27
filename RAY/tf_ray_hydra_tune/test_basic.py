import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from modules import train_model

@hydra.main(version_base=None, config_path="config", config_name="config")
def test_basic(cfg: DictConfig):
    print("Configuration loaded successfully!")
    print(f"Model config: {cfg}")
    
    # Test basic model training without Ray Tune
    print("\nStarting basic model training...")
    model = train_model(cfg)
    print("Training completed successfully!")

if __name__ == "__main__":
    test_basic()
