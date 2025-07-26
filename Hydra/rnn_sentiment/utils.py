import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def build_vocab(path):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    vocab = set()
    
    for text in texts:
        words = text.lower().split() # Split text into words and convert to lowercase
        vocab.update(words)
    
    # Add padding token and unknown token
    vocab_list = ['<PAD>', '<UNK>'] + list(vocab)
    return {word: idx for idx, word in enumerate(vocab_list)}

def encode_sentence(sentence, vocab, max_length=None):
    words = sentence.lower().split()
    # Convert words to indices
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad sequences if max_length is specified
    if max_length:
        if len(indices) < max_length:
            indices += [vocab['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
    
    return np.array(indices)

def load_dataset(path, vocab, max_seq_len=5):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Encode texts to sequences of indices
    encoded_texts = [encode_sentence(text, vocab, max_seq_len) for text in texts]
    X = np.array(encoded_texts)
    
    # Convert to one-hot encoding
    vocab_size = len(vocab)
    X_onehot = np.zeros((X.shape[0], X.shape[1], vocab_size))
    
    for i, sequence in enumerate(X):
        for j, word_idx in enumerate(sequence):
            X_onehot[i, j, word_idx] = 1
    
    # Convert labels to categorical (one-hot) if needed
    y = np.array(labels)
    
    return X_onehot, y