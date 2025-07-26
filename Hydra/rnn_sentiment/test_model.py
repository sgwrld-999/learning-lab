import tensorflow as tf
import numpy as np
from utils import build_vocab, encode_sentence

# Load the trained model
model = tf.keras.models.load_model('rnn_model.h5')

# Load the vocabulary
vocab = build_vocab('data/sentiment.csv')
print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary: {vocab}")

# Test with a new sentence
test_sentence = "you are good"
print(f"\nTesting sentence: '{test_sentence}'")

# Encode the sentence
encoded = encode_sentence(test_sentence, vocab, max_length=5)
print(f"Encoded indices: {encoded}")

# Convert to one-hot encoding
vocab_size = len(vocab)
X_test = np.zeros((1, 5, vocab_size))  # batch_size=1, seq_len=5, vocab_size
for j, word_idx in enumerate(encoded):
    X_test[0, j, word_idx] = 1

print(f"One-hot encoded shape: {X_test.shape}")
print(f"One-hot encoded (first word): {X_test[0, 0, :]}")

# Make prediction
prediction = model.predict(X_test, verbose=0)
print(f"\nPrediction probability: {prediction[0][0]:.4f}")
print(f"Predicted sentiment: {'Positive' if prediction[0][0] > 0.5 else 'Negative'}")

# Test with another sentence
test_sentence2 = "you are bad"
print(f"\nTesting sentence: '{test_sentence2}'")

encoded2 = encode_sentence(test_sentence2, vocab, max_length=5)
X_test2 = np.zeros((1, 5, vocab_size))
for j, word_idx in enumerate(encoded2):
    X_test2[0, j, word_idx] = 1

prediction2 = model.predict(X_test2, verbose=0)
print(f"Prediction probability: {prediction2[0][0]:.4f}")
print(f"Predicted sentiment: {'Positive' if prediction2[0][0] > 0.5 else 'Negative'}")
