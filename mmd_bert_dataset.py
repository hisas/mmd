import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertJapaneseTokenizer

tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
max_len = 50

def convert_sentence_to_ids(sentence):
    sentence = '[CLS] ' + sentence + ' [SEP]'
    sentence_tokens = tokenizer.tokenize(sentence)
    sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    sentence_len = len(sentence_ids)
    if sentence_len < max_len + 1:
        sentence_ids += [0] * (max_len - sentence_len)

    return torch.tensor(sentence_ids, dtype=torch.long), sentence_len


def load_ids_and_labels(row, task):
    context, context_len = convert_sentence_to_ids(row['Context'])

    image_path = row['Image']
    gaze = torch.tensor([row['EyeX'], row['EyeY'], row['GazeX'], row['GazeY']])

    response = row['Action'] if task == 'action' else row['Response']
    response, response_len = convert_sentence_to_ids(response)

    label = torch.tensor([row['Label']], dtype=torch.float32)

    return context, context_len, image_path, gaze, response, response_len, label

def test_load_ids(i, row):
    context, context_len = convert_sentence_to_ids(row['Context'])

    image_path = row['Image']
    gaze = torch.tensor([row['EyeX'], row['EyeY'], row['GazeX'], row['GazeY']])

    response, response_len = convert_sentence_to_ids(row[i])

    return context, context_len, image_path, gaze, response, response_len

class MmdBertDataset(Dataset):
    def __init__(self, path, task, train):
        self.crl = []
        for _, row in pd.read_csv(path).iterrows():
            if 'test_10.csv' in path:
                for i in range(6, 16):
                    self.crl.append(test_load_ids(i, row))
            else:
                self.crl.append(load_ids_and_labels(row, task))

    def __len__(self):
        return len(self.crl)

    def __getitem__(self, idx):
        return self.crl[idx]
