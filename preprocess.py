import os
import pickle
import random
import subprocess
import zipfile
from urllib.request import urlretrieve

import pandas as pd

from mmd_bert_dataset import *
from mmd_dataset import *

random.seed(0)

def extract_gaze(df):
    df_ann = pd.read_csv('data/data_new/train_annotations.txt', delimiter=',', header=None)
    gaze = {}
    for _, r in df_ann.iterrows():
        gaze[r[1]] = [r[8], r[9]]

    data = []
    for _, r in df.iterrows():
        data.append([r[0], r[1], gaze[r[2]][0], gaze[r[2]][1], r[3]])

    return pd.DataFrame(data)

def remove_x(df, task):
    ng_words = ['なし', 'なi', '罰', 'ばつ', 'バツ', 'batu ', 'b', 'u', 'v', 'z']
    if task == 'action':
        ng_words.append('はい')
    df = df[~df[task].isin(ng_words)]

    df = df[~df[task].str.contains('×')]
    df = df[~df[task].str.contains('x')]
    df = df[~df[task].str.contains('✖')]
    df = df[~df[task].str.contains('ｘ')]
    df = df[~df[task].str.contains('Ｘ')]
    df = df[~df[task].str.contains('ⅹ')]
    df = df[~df[task].str.contains('✕')]
    df = df[~df[task].str.contains('X')]

    if task == 'action':
        df = df[~df[task].str.contains('http')]
        df = df[~df[task].str.contains('Heydouga')]
        df = df[~df[task].str.contains('楽天Beauty')]
        df = df[~df[task].str.contains('何もしな')]
        df = df[~df[task].str.contains('なにもしな')]
        df = df[~df[task].str.contains('特にな')]
        df = df[~df[task].str.contains('とくにな')]

    return df

def filter_response_df(path):
    df = pd.read_csv(path, delimiter='\t', usecols=['text1', 'image', 'gf_id', 'response'])

    df = remove_x(df, 'response')
    df['text1'] = df['text1'].str[56:-52].str.strip()
    df = df[~df['text1'].isin(['""""""', '##', '$$$'])]
    df['response'] = df['response'].str.strip()

    train_df, val_df, test_df = df[:len(df)-24000], df[len(df)-24000:len(df)-12000], df[len(df)-12000:len(df)]
    train_df, val_df, test_df = extract_gaze(train_df), extract_gaze(val_df), extract_gaze(test_df)

    return train_df, val_df, test_df

def export_response_df(df, set):
    context_response = []
    for _, r in df.iterrows():
        context_response.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Response', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_response)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/response_' + set + '.csv', index=False, header=False)

def export_response_test_df(df):
    context_response = []
    for _, r in df.iterrows():
        context_response.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Response', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_response)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/response_test_1.csv', index=False, header=False)

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Ground Truth Response', 'Distractor_0', 'Distractor_1', \
             'Distractor_2', 'Distractor_3', 'Distractor_4', 'Distractor_5', 'Distractor_6', 'Distractor_7', 'Distractor_8']]

    for _, r in df.iterrows():
        while True:
            ts = random.sample(context_response, 9)
            flag = True
            ds = []
            for t in ts:
                if t[0] != r[0]:
                    ds.append(t[1])
                else:
                    flag = False

            if flag == True:
                data.append(sum([[r[0]], [r[1]], [r[2]], [r[3]], [r[4]], ds], []))
                break
    pd.DataFrame(data).to_csv('data/response_test_10.csv', index=False, header=False)

def filter_action_df(path):
    df = pd.read_csv(path, delimiter='\t', usecols=['text1', 'image', 'gf_id', 'action'])

    df = remove_x(df, 'action')
    df['text1'] = df['text1'].str[56:-52].str.strip()
    df = df[~df['text1'].isin(['""""""', '##', '$$$'])]
    df['action'] = df['action'].str.strip()

    train_df, val_df, test_df = df[:len(df)-6000], df[len(df)-6000:len(df)-3000], df[len(df)-3000:len(df)]
    train_df, val_df, test_df = extract_gaze(train_df), extract_gaze(val_df), extract_gaze(test_df)

    return train_df, val_df, test_df

def export_action_df(df, set):
    context_action = []
    for _, r in df.iterrows():
        context_action.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Action', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_action)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/action_' + set + '.csv', index=False, header=False)

def export_action_test_df(df):
    context_action = []
    for _, r in df.iterrows():
        context_action.append([r[0], r[4]])

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Action', 'Label']]
    for _, r in df.iterrows():
        data.append([r[0], r[1], r[2], r[3], r[4], 1])

        while True:
            t = random.choice(context_action)
            if t[0] != r[0]:
                data.append([r[0], r[1], r[2], r[3], t[1], 0])
                break

    pd.DataFrame(data).to_csv('data/action_test_1.csv', index=False, header=False)

    data = [['Context', 'Image', 'GazeX', 'GazeY', 'Ground Truth Action', 'Distractor_0', 'Distractor_1', \
             'Distractor_2', 'Distractor_3', 'Distractor_4', 'Distractor_5', 'Distractor_6', 'Distractor_7', 'Distractor_8']]

    for _, r in df.iterrows():
        while True:
            ts = random.sample(context_action, 9)
            flag = True
            ds = []
            for t in ts:
                if t[0] != r[0]:
                    ds.append(t[1])
                else:
                    flag = False

            if flag == True:
                data.append(sum([[r[0]], [r[1]], [r[2]], [r[3]], [r[4]], ds], []))
                break
    pd.DataFrame(data).to_csv('data/action_test_10.csv', index=False, header=False)

print('Downloading file')
os.mkdir('data')
urlretrieve('http://gazefollow.csail.mit.edu/downloads/data.zip', 'data/data.zip')
subprocess.call('unzip -q data/data.zip -d data', shell=True)
os.remove('data/data.zip')
urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz', 'data/cc.ja.300.vec.gz')
subprocess.call('gzip data/cc.ja.300.vec.gz -cd > data/cc.ja.300.vec', shell=True)
os.remove('data/cc.ja.300.vec.gz')

print('Exporting response_train.csv, response_val.csv, response_test_1.csv and response_test_10.csv')
train_response_df, val_response_df, test_response_df = filter_response_df('original_data/jparvsu-response.tsv')
export_response_df(train_response_df, 'train')
export_response_df(val_response_df, 'val')
export_response_test_df(test_response_df)

print('Exporting action_train.csv, action_val.csv, action_test_1.csv and action_test_10.csv')
train_action_df, val_action_df, test_action_df = filter_action_df('original_data/jparvsu-response.tsv')
export_action_df(train_action_df, 'train')
export_action_df(val_action_df, 'val')
export_action_test_df(test_action_df)

print('Creating response_dataset.pkl')
max_len = 40
params = {}
params['train'] = MmdDataset('data/response_train.csv', task='response', train=True, max_len=max_len)
params['val'] = MmdDataset('data/response_val.csv', task='response', train=False, max_len=max_len)
params['test_1'] = MmdDataset('data/response_test_1.csv', task='response', train=False, max_len=max_len)
params['test_10'] = MmdDataset('data/response_test_10.csv', task='response', train=False, max_len=max_len)
with open('data/response_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)

print('Creating action_dataset.pkl')
params = {}
params['train'] = MmdDataset('data/action_train.csv', task='action', train=True, max_len=max_len)
params['val'] = MmdDataset('data/action_val.csv', task='action', train=False, max_len=max_len)
params['test_1'] = MmdDataset('data/action_test_1.csv', task='action', train=False, max_len=max_len)
params['test_10'] = MmdDataset('data/action_test_10.csv', task='action', train=False, max_len=max_len)
with open('data/action_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)

print('Creating response_bert_dataset.pkl')
params = {}
params['train'] = MmdBertDataset('data/response_train.csv', task='response', train=True)
params['val'] = MmdBertDataset('data/response_val.csv', task='response', train=False)
params['test_1'] = MmdBertDataset('data/response_test_1.csv', task='response', train=False)
params['test_10'] = MmdBertDataset('data/response_test_10.csv', task='response', train=False)
with open('data/response_bert_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)

print('Creating action_bert_dataset.pkl')
params = {}
params['train'] = MmdBertDataset('data/action_train.csv', task='action', train=True)
params['val'] = MmdBertDataset('data/action_val.csv', task='action', train=False)
params['test_1'] = MmdBertDataset('data/action_test_1.csv', task='action', train=False)
params['test_10'] = MmdBertDataset('data/action_test_10.csv', task='action', train=False)
with open('data/action_bert_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)
