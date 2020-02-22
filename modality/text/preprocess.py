import os
import pickle
from dataset import *

if not os.path.exists('pkl'):
    os.makedirs('pkl')

data_dir = '../../data/'

max_len = 40

params = {}
params['train'] = TextDataset(data_dir + 'response_train.csv', task='response', train=True, max_len=max_len)
params['val'] = TextDataset(data_dir + 'response_val.csv', task='response', train=False, max_len=max_len)
params['test_1'] = TextDataset(data_dir + 'response_test_1.csv', task='response', train=False, max_len=max_len)
params['test_10'] = TextDataset(data_dir + 'response_test_10.csv', task='response', train=False, max_len=max_len)
with open('pkl/response_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)


params = {}
params['train'] = TextDataset(data_dir + 'action_train.csv', task='action', train=True, max_len=max_len)
params['val'] = TextDataset(data_dir + 'action_val.csv', task='action', train=False, max_len=max_len)
params['test_1'] = TextDataset(data_dir + 'action_test_1.csv', task='action', train=False, max_len=max_len)
params['test_10'] = TextDataset(data_dir + 'action_test_10.csv', task='action', train=False, max_len=max_len)
with open('pkl/action_dataset.pkl', 'wb') as f:
    pickle.dump(params, f, -1)
