import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from modules import train_model, get_model, load_model
from ray import tune

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Initialize Ray Tune
    tune.run(
        train_model,
        config={
            'input_length': cfg['input_length'],
            'input_dim': cfg['input_dim'],
            'hidden_units': cfg['hidden_units'],
            'num_layers': cfg['num_layers'],
            'output_dim': cfg['output_dim'],
            'optimizer': cfg['optimizer'],
            'loss': cfg['loss'],
            'epochs': cfg['epochs'],
            'batch_size': cfg['batch_size'],
            'model_save_path': to_absolute_path(cfg['model_save_path'])
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        }
    )

if __name__ == "__main__":
    main()