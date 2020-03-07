import os
import re

import MeCab
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from torch.utils.data import Dataset


def create_word_to_id_and_id_to_word(df, task):
    word_to_id = {'<PAD>': 0, '<UNK>': 1}
    id_to_word = {0: '<PAD>', 1: '<UNK>'}

    for _, r in df.iterrows():
        if task == 'response':
            sentences = [r['Context'], r['Response']]
        elif task == 'action':
            sentences = [r['Context'], r['Action']]

        for sentence in sentences:
            sentence_words = sentence2words(sentence)
            for word in sentence_words:
                if word not in word_to_id:
                    l = len(word_to_id)
                    word_to_id[word] = l
                    id_to_word[l] = word

    return word_to_id, id_to_word

def preprocessing_text(text):
    text = re.sub('#', '', text)

    return text

def sentence2words(sentence):
    mecab = MeCab.Tagger()
    sentence_words = []
    sentence = preprocessing_text(sentence)
    for m in mecab.parse(sentence).split("\n"):
        w = m.split("\t")[0].lower()
        if len(w) == 0 or w == 'eos':
            continue
        sentence_words.append(w)
    return sentence_words

def create_id_to_vec(word_to_id):
    model = KeyedVectors.load_word2vec_format('data/cc.ja.300.vec', binary=False)
    id_to_vec = {}

    for word, id in word_to_id.items():
        try:
            id_to_vec[word_to_id[word]] = torch.tensor(model[word], dtype=torch.float32)
        except KeyError:
            v = np.zeros(300, dtype='float32')
            v[:] = np.random.randn(*v.shape)*0.01
            id_to_vec[word_to_id[word]] = torch.tensor(v, dtype=torch.float32)

    embedding_dim = id_to_vec[0].shape[0]

    return id_to_vec, embedding_dim

def load_ids_and_labels(row, word_to_id, max_len, task):
    context_ids = []
    for word in sentence2words(row['Context']):
        if word in word_to_id:
            context_ids.append(word_to_id[word])
        else:
            context_ids.append(1)
    context_len = len(context_ids)
    if context_len < max_len + 1:
        context_ids += [0] * (max_len - context_len)
    context = torch.tensor(context_ids, dtype=torch.long)

    image_path = row['Image']
    gaze = torch.tensor([row['GazeX'], row['GazeY']])

    response_ids = []
    sentence = row['Action'] if task == 'action' else row['Response']
    for word in sentence2words(sentence):
        if word in word_to_id:
            response_ids.append(word_to_id[word])
        else:
            response_ids.append(1)
    response_len = len(response_ids)
    if response_len < max_len + 1:
        response_ids += [0] * (max_len - response_len)
    response = torch.tensor(response_ids, dtype=torch.long)

    label = torch.tensor([row['Label']], dtype=torch.float32)

    return context, context_len, image_path, gaze, response, response_len, label

def test_load_ids(i, row, word_to_id, max_len):
    context_ids = []
    for word in sentence2words(row['Context']):
        if word in word_to_id:
            context_ids.append(word_to_id[word])
        else:
            context_ids.append(1)
    context_len = len(context_ids)
    if context_len < max_len + 1:
        context_ids += [0] * (max_len - context_len)
    context = torch.tensor(context_ids, dtype=torch.long)

    image_path = row['Image']
    gaze = torch.tensor([row['GazeX'], row['GazeY']])

    response_ids = []
    for word in sentence2words(row[i]):
        if word in word_to_id:
            response_ids.append(word_to_id[word])
        else:
            response_ids.append(1)
    response_len = len(response_ids)
    if response_len < max_len + 1:
        response_ids += [0] * (max_len - response_len)
    response = torch.tensor(response_ids, dtype=torch.long)

    return context, context_len, image_path, gaze, response, response_len

class MmdDataset(Dataset):
    def __init__(self, path, task, train, max_len):
        df = pd.read_csv(path)
        if train:
            global word_to_id, id_to_word
            word_to_id, id_to_word = create_word_to_id_and_id_to_word(df, task)
            self.word_to_id = word_to_id
            self.id_to_word = id_to_word
            self.id_to_vec, self.emb_dim = create_id_to_vec(word_to_id)

        self.crl = []
        for _, row in df.iterrows():
            if 'test_10.csv' in path:
                for i in range(4, 14):
                    self.crl.append(test_load_ids(i, row, word_to_id, max_len))
            else:
                self.crl.append(load_ids_and_labels(row, word_to_id, max_len, task))

    def __len__(self):
        return len(self.crl)

    def __getitem__(self, idx):
        return self.crl[idx]

