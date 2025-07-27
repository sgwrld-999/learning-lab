import tensorflow as tf
from tensorflow.keras import models, layers
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tensorflow.keras.datasets import mnist


@hydra.main(version_base=None, config_path="config", config_name="config")
def get_model(cfg):
    model = models.Sequential()
    #input layer
    model.add(layers.Input(shape=(cfg['input_length'], cfg['input_dim'])))
    #hidden layers
    for i in range(cfg['num_layers']):
        # Only return sequences for layers except the last one
        return_sequences = (i < cfg['num_layers'] - 1)
        model.add(layers.LSTM(cfg['hidden_units'], return_sequences=return_sequences))
    #output layer
    model.add(layers.Dense(cfg['output_dim'], activation='softmax'))

    #compile model
    model.compile(optimizer=cfg['optimizer'], loss=cfg['loss'], metrics=['accuracy'])
    #save model configuration
    model.save(to_absolute_path(cfg['model_save_path']))    
    return model

def load_model(model_path):
    return models.load_model(to_absolute_path(model_path))

def train_model(config):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], -1, config['input_dim']))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=config['output_dim'])
    model = get_model(config)
    model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'])
    model.save(to_absolute_path(config['model_save_path'])) 
    
    loss, acc = model.evaluate(x_train, y_train)
    print(f"Training completed with loss: {loss}, accuracy: {acc}")
    return model