import hydra
import os
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from models.rnn import build_rnn_model
from utils import build_vocab, load_dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    print("Training configuration:", OmegaConf.to_yaml(cfg))
    vocab = build_vocab(cfg.data.data.path)
    X_train, y_train = load_dataset(cfg.data.data.path, vocab, cfg.data.data.max_seq_len)
    
    # Update input_dim to match vocabulary size
    model = build_rnn_model(
        input_dim=len(vocab),  # Use vocabulary size for one-hot encoding
        hidden_size=cfg.model.model.hidden_size,
        num_layers=cfg.model.model.num_layers,
        output_dim=cfg.model.model.output_dim,
        activation=cfg.model.model.activation
    )
    
    model.fit(X_train, y_train, epochs=cfg.train.train.epochs, batch_size=cfg.train.train.batch_size)
    
    model.save("rnn_model.h5")
    
if __name__ == "__main__":
    train()