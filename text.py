'''
Module to preprocess filckr8k text data
'''
import os
import string
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

def load_token_text(token_dir):

    sents_dict = {}
    with open(token_dir, 'r') as f:
        for line in f.readlines():
            words = line.strip('\n').split()
            img_id = words[0].split('.')[0]
            sent = ' '.join(words[1:])

            if img_id in sents_dict.keys():
                sents_dict[img_id].append(sent)
            else:
                sents_dict[img_id] = [sent]
            
    return sents_dict


def load_dataset_token(dataset_dir, token_dir, start_end = True):
    
    all_sents = load_token_text(token_dir)

    img_ids = []
    with open(dataset_dir, 'r') as f:
        for line in f.readlines():
            img_ids.append(os.path.splitext(line)[0])

    sent_list = []
    for id in img_ids:
        for sent in all_sents[id]:
            sent_ = sent
            if start_end:
                sent_ = 'startseq ' + sent_ + ' endseq'

            sent_list.append(sent_)
    
    return sent_list


def create_tokenizer(dataset_dir, token_dir, start_end = True, use_all = False):

    # 'num_words = None' for all words in training set
    # for example, 'num_words = 6000', means use maximum 6000 words in vocabulary  
    num_words = None

    sent_list = load_dataset_token(dataset_dir, token_dir, start_end)

    if use_all:
        tokenizer = Tokenizer()
    else:
        if num_words:
            tokenizer = Tokenizer(num_words)
        else:
            tokenizer = Tokenizer()

    tokenizer.fit_on_texts(sent_list)

    return tokenizer


def clean_test_sentences(tokenizer, sents_list):

    cleaned_sents_list= []
    for sents in sents_list:
        sequences = tokenizer.texts_to_sequences(sents)
        cleaned_sents_list.append(tokenizer.sequences_to_texts(sequences))
    
    return cleaned_sents_list
