import os
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../..')
import argparse
import json
from attrdict import AttrDict
import pickle
from datetime import datetime
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from helper.image_helper import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path')
parser.add_argument('--log', action='store_true')
args = parser.parse_args()
path = args.path
log = args.log
task = path.split('/')[1] 
text_model = path.split('/')[2].split('_')[0]
image_model = path.split('/')[2].split('_')[1] 
synthesis_method = path.split('/')[2].split('_')[2]

if log:
    save_dir = 'log/' + task
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    format = '%(message)s'
    filename = save_dir + '/' + text_model + '_' + image_model + '_' \
               + synthesis_method + '_' + datetime.now().strftime('%Y%m%d%H%M') + '.log'
    logging.basicConfig(filename=filename, level=logging.DEBUG, format=format)


def get_config(file_path):
    config_file = file_path
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)
    config = AttrDict(json_object)

    return config

def h(sentence_ids):
    ans = ''
    for it in sentence_ids:
        i = it.item()
        if i == 0:
            break
        else:
            ans += train.id_to_word[i]

    return ans

def calc_test_accuracy():
    test_correct_count = 0

    for contexts, contexts_len, images_path, gazes, responses, responses_len, labels in tqdm(test_1_loader):
        with torch.no_grad():
            images = fetch_images(images_path, 'train')
            images = images.to(device)
            gazes = gazes.to(device)

            if text_model == 'lstm':
                contexts_len, csi = contexts_len.sort(descending=True)
                sorted_contexts = contexts[csi]
                responses_len, rsi = responses_len.sort(descending=True)
                sorted_responses = responses[rsi]
                sorted_contexts = sorted_contexts.to(device)
                sorted_responses = sorted_responses.to(device)
                labels = labels.to(device)
                probs = encoder(sorted_contexts, contexts_len, csi, images, gazes, sorted_responses, responses_len, rsi)
            elif text_model == 'transformer':
                contexts, responses, labels = contexts.to(device), responses.to(device), labels.to(device)
                cm, rm = (contexts != 0).unsqueeze(-2), (responses != 0).unsqueeze(-2)
                probs = encoder(contexts, cm, images, gazes, responses, rm)

            for i, (prob, label) in enumerate(zip(probs, labels)):
                p, l, g1, g2 = prob.item(), label.item(), gazes[i][0].item(), gazes[i][1].item()
                if ((p >= 0.5) and (l == 1.0)) or ((p < 0.5) and (l == 0.0)):
                    test_correct_count += 1
                    logging.info('%s %s %s %s %s %s %s %s', 'o', h(contexts[i]), h(responses[i]), images_path[i], g1, g2, int(l), round(p, 2))
                else:
                    logging.info('%s %s %s %s %s %s %s %s', 'x', h(contexts[i]), h(responses[i]), images_path[i], g1, g2, int(l), round(p, 2))

    test_accuracy = test_correct_count / len(test_1)

    return test_accuracy

def get_recall_at_k():
    recall_at_5_true_hits, recall_at_2_true_hits, recall_at_1_true_hits = 0, 0, 0

    for contexts, contexts_len, images_path, gazes, responses, responses_len in tqdm(test_10_loader):
        with torch.no_grad():
            images = fetch_images(images_path, 'train')
            images = images.to(device)
            gazes = gazes.to(device)

            if text_model == 'lstm':
                contexts_len, csi = contexts_len.sort(descending=True)
                sorted_contexts = contexts[csi]
                responses_len, rsi = responses_len.sort(descending=True)
                sorted_responses = responses[rsi]
                sorted_contexts = sorted_contexts.to(device)
                sorted_responses = sorted_responses.to(device)
                probs = encoder(sorted_contexts, contexts_len, csi, images, gazes, sorted_responses, responses_len, rsi)
            elif text_model == 'transformer':
                contexts, responses = contexts.to(device), responses.to(device)
                cm, rm = (contexts != 0).unsqueeze(-2), (responses != 0).unsqueeze(-2)
                probs = encoder(contexts, cm, images, gazes, responses, rm)
            
            msg = ''
            sorted_probs, sorted_idx = torch.sort(probs, dim=0, descending=True)

            if probs[0] in sorted_probs[:5]:
                recall_at_5_true_hits += 1
                msg += 'o'
            else:
                msg += 'x'

            if probs[0] in sorted_probs[:2]:
                recall_at_2_true_hits += 1
                msg += 'o'
            else:
                msg += 'x'

            if probs[0] in sorted_probs[:1]:
                recall_at_1_true_hits += 1
                msg += 'o'
            else:
                msg += 'x'

            logging.info('%s %s %s %s %s %s', msg, h(contexts[0]), h(responses[0]), images_path[0], gazes[0][0].item(), gazes[0][1].item())
            for r, p in zip(responses[sorted_idx], sorted_probs):
                logging.info('%s %s %s', '\t', h(r[0]), p.item())

    recall_at_5 = recall_at_5_true_hits / (len(test_10) / 10)
    recall_at_2 = recall_at_2_true_hits / (len(test_10) / 10)
    recall_at_1 = recall_at_1_true_hits / (len(test_10) / 10)

    return recall_at_5, recall_at_2, recall_at_1


with open('../../data/' + task + '_dataset.pkl', 'rb') as f:
    params = pickle.load(f)
    train = params['train']
    test_1 = params['test_1']
    test_10 = params['test_10']

id_to_vec = train.id_to_vec
emb_size = train.emb_dim
vocab_size = len(train.word_to_id)
if text_model == 'lstm':
    config = get_config('config/lstm_config.json')
    from model.text_image_gaze_encoder import TextImageGazeLstmEncoder
    encoder = TextImageGazeLstmEncoder(image_model, synthesis_method, id_to_vec, emb_size, vocab_size, config)
elif text_model == 'transformer':
    config = get_config('config/transformer_config.json')
    from model.text_image_gaze_encoder import TextImageGazeTransformerEncoder
    encoder = TextImageGazeTransformerEncoder(image_model, synthesis_method, id_to_vec, emb_size, vocab_size, config, device) 
encoder.load_state_dict(torch.load(path))
encoder.to(device)
encoder.eval()

test_1_loader = DataLoader(test_1, batch_size=64, shuffle=False, num_workers=4)
print("Test accuracy", calc_test_accuracy())

logging.info('\n')

test_10_loader = DataLoader(test_10, batch_size=10, shuffle=False, num_workers=4)
print("Recall at k", get_recall_at_k())
