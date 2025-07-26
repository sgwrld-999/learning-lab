import tensorflow as tf
from tensorflow.keras import layers, models


def build_rnn_model(input_dim, hidden_size, num_layers, output_dim, activation):
    model = models.Sequential()
    
    for i in range(num_layers):
        return_sequences = i < num_layers - 1 # Ensure the last layer does not return sequences, to stop the recurrent connections
        model.add(layers.SimpleRNN(hidden_size, return_sequences=return_sequences, input_shape=(None, input_dim) if i == 0 else None))
        
    model.add(layers.Dense(output_dim, activation=activation))
    
    # Use binary crossentropy for binary classification with sigmoid activation
    if output_dim == 1 and activation == 'sigmoid':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    