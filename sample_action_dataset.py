import pickle

from mmd_bert_dataset import *
from mmd_dataset import *


max_len = 40
params = {}
params['train'] = MmdDataset('data/action_train.csv', task='action', train=True, max_len=max_len)
params['test_10'] = MmdDataset('data/action_sample_test_10.csv', task='action', train=False, max_len=max_len)
with open('data/response_sample_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)

params['train'] = MmdBertDataset('data/action_train.csv', task='action', train=True)
params['test_10'] = MmdBertDataset('data/action_sample_test_10.csv', task='action', train=False)
with open('data/action_sample_bert_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)
