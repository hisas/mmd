import argparse
import json
import logging
import os
import pathlib
import pickle
import re
import sys
from datetime import datetime

import torch
from attrdict import AttrDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertJapaneseTokenizer

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../..')

from helper.image_helper import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path')
args = parser.parse_args()
path = args.path
task = path.split('/')[1] 
text_model = path.split('/')[2].split('_')[0]
image_model = path.split('/')[2].split('_')[1] 
fusion_method = path.split('/')[2].split('_')[2]

tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')

def set_log():
    save_dir = 'log/sample_' + task
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    format = '%(message)s'
    filename = save_dir + '/' + text_model + '_' + image_model + '_' \
               + fusion_method + '_' + datetime.now().strftime('%Y%m%d%H%M') + '.txt'
    logging.basicConfig(filename=filename, level=logging.DEBUG, format=format)

def get_config(file_path):
    config_file = file_path
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)
    config = AttrDict(json_object)

    return config

def h(sentence_ids):
    if text_model == 'bert':
        tokens = tokenizer.convert_ids_to_tokens(sentence_ids)
        remove_tokens = ['[CLS]', '[PAD]', '[SEP]']
        sentence = ''.join([t for t in tokens if t not in(['[CLS]', '[PAD]', '[SEP]'])])
        sentence = re.sub(r'##', '', sentence)
    else:
        sentence = ''
        for si in sentence_ids:
            i = si.item()
            if i == 0:
                break
            else:
                sentence += train.id_to_word[i]

    return sentence

def get_recall_at_k():
    for contexts, contexts_len, images_path, _, responses, responses_len in tqdm(test_10_loader):
        with torch.no_grad():
            images = fetch_images(images_path, 'train')
            images = images.to(device)

            if text_model == 'lstm':
                contexts_len, csi = contexts_len.sort(descending=True)
                sorted_contexts = contexts[csi]
                responses_len, rsi = responses_len.sort(descending=True)
                sorted_responses = responses[rsi]
                sorted_contexts = sorted_contexts.to(device)
                sorted_responses = sorted_responses.to(device)
                probs = encoder(sorted_contexts, contexts_len, csi, images, sorted_responses, responses_len, rsi)
            elif text_model == 'transformer':
                contexts, responses = contexts.to(device), responses.to(device)
                cm, rm = (contexts != 0).unsqueeze(-2), (responses != 0).unsqueeze(-2)
                probs = encoder(contexts, cm, images, responses, rm)
            elif text_model == 'bert':
                contexts, responses = contexts.to(device), responses.to(device)
                cm, rm = (contexts != 0).int(), (responses != 0).int()
                probs = encoder(contexts, cm, images, responses, rm)

            msg = ''
            sorted_probs, sorted_idx = torch.sort(probs, dim=0, descending=True)

            if probs[0] in sorted_probs[:5]:
                msg += 'o'
            else:
                msg += 'x'

            if probs[0] in sorted_probs[:2]:
                msg += 'o'
            else:
                msg += 'x'

            if probs[0] in sorted_probs[:1]:
                msg += 'o'
            else:
                msg += 'x'

            logging.info('%s %s %s %s', msg, h(contexts[0]), h(responses[0]), images_path[0])
            for r, p in zip(responses[sorted_idx], sorted_probs):
                logging.info('%s %s %s', '\t', h(r[0]), p.item())


if text_model == 'bert':
    data_path = '../../data/' + task + '_sample_bert_dataset.pkl'
else:
    data_path = '../../data/' + task + '_sample_dataset.pkl'
with open(data_path, 'rb') as f:
    params = pickle.load(f)
    train = params['train']
    test_10 = params['test_10']

if text_model != 'bert':
    id_to_vec = train.id_to_vec
    emb_size = train.emb_dim
    vocab_size = len(train.word_to_id)

config = get_config('config/' + text_model + '_config.json')
if text_model == 'lstm':
    from model.text_image_encoder import TextImageLstmEncoder
    encoder = TextImageLstmEncoder(image_model, fusion_method, id_to_vec, emb_size, vocab_size, config)
elif text_model == 'transformer':
    from model.text_image_encoder import TextImageTransformerEncoder
    encoder = TextImageTransformerEncoder(image_model, fusion_method, id_to_vec, emb_size, vocab_size, config, device)
elif text_model == 'bert':
    from model.text_image_encoder import TextImageBertEncoder
    encoder = TextImageBertEncoder(image_model, fusion_method, config)
encoder.load_state_dict(torch.load(path))
encoder.to(device)
encoder.eval()

set_log()

test_10_loader = DataLoader(test_10, batch_size=10, shuffle=False, num_workers=4)
get_recall_at_k()
