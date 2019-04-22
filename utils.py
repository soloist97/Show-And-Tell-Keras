'''
File to define data generator for training
'''

import numpy as np
from keras.utils import to_categorical

from preprocessing.image import load_features
from preprocessing.text import load_dataset_token
from NIC import unit_size

def batch_generator(batch_size, max_len, tokenizer, dict_dir, dataset_dir, token_dir):

    vocab_size = tokenizer.num_words or (len(tokenizer.word_index)+1)

    img_features = load_features(dict_dir, dataset_dir, 5)
    raw_sentences = load_dataset_token(dataset_dir, token_dir, True)

    N = img_features.shape[0]
    
    while True:
        for i in range(0, N, batch_size):

            sequences = tokenizer.texts_to_sequences(raw_sentences[i:i+batch_size])
                
            X_text = []
            Y_text = []
            for seq in sequences:
                if len(seq) > max_len:
                    X_text.append(seq[:max_len])
                    Y_text.append(seq[1:max_len+1])
                else:
                    X_text.append(seq[:len(seq)-1] + [0]*(max_len-len(seq)+1))
                    Y_text.append(seq[1:] + [0]*(max_len-len(seq)+1))

            X_text_mat = np.array(X_text)
            Y_text_mat = to_categorical(Y_text, vocab_size)

            yield ([img_features[i:i+batch_size, :], X_text_mat, np.zeros([X_text_mat.shape[0], unit_size]), np.zeros([X_text_mat.shape[0], unit_size])], 
                    Y_text_mat)

