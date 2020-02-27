import argparse
import json
import os
import pathlib
import pickle
import random
import sys
from datetime import datetime

import numpy as np
import torch
from attrdict import AttrDict
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from helper.image_helper import *
from mmd_dataset import MmdDataset

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../..')

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--task')
parser.add_argument('--text_model')
parser.add_argument('--image_model')
args = parser.parse_args()
task = args.task
text_model = args.text_model
image_model = args.image_model


def get_config(file_path):
    config_file = file_path
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)
    config = AttrDict(json_object)

    return config

def train_model(learning_rate, l2_penalty, epochs):
    print(str(datetime.now()).split('.')[0], "Starting training and validation...\n")

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    criterion = nn.BCELoss()
    best_val_accuracy = 0.0

    for epoch in range(1, epochs+1):
        sum_loss_training = 0.0
        training_correct_count = 0
        encoder.train()

        for _, _, images_path, _, responses, responses_len, labels in tqdm(train_loader):
            encoder.zero_grad()

            images = fetch_images(images_path, 'train')
            images = images.to(device)

            if text_model == 'lstm':
                responses_len, rsi = responses_len.sort(descending=True)
                sorted_responses = responses[rsi]
                sorted_responses = sorted_responses.to(device)
                labels = labels.to(device)
                probs = encoder(images, sorted_responses, responses_len, rsi)
            elif text_model == 'transformer':
                responses, labels = responses.to(device), labels.to(device)
                rm = (responses != 0).unsqueeze(-2)
                probs = encoder(images, responses, rm)
            elif text_model == 'bert':
                responses, labels = responses.to(device), labels.to(device)
                rm = (responses != 0).int()
                probs = encoder(images, responses, rm)

            loss = criterion(probs, labels)
            sum_loss_training += loss.item()
            loss.backward()
            optimizer.step()
            for prob, label in zip(probs, labels):
               p, l = prob.item(), label.item() 
               if ((p >= 0.5) and (l == 1.0)) or ((p < 0.5) and (l == 0.0)):
                   training_correct_count += 1

        training_accuracy = training_correct_count / len(train)

        val_correct_count = 0
        sum_loss_val = 0.0
        encoder.eval()

        for _, _, images_path, _, responses, responses_len, labels in tqdm(val_loader):
            with torch.no_grad():
                images = fetch_images(images_path, 'test')
                images = images.to(device)

                if text_model == 'lstm':
                    responses_len, rsi = responses_len.sort(descending=True)
                    sorted_responses = responses[rsi]
                    sorted_responses = sorted_responses.to(device)
                    labels = labels.to(device)
                    probs = encoder(images, sorted_responses, responses_len, rsi)
                elif text_model == 'transformer':
                    responses, labels = responses.to(device), labels.to(device)
                    rm = (responses != 0).unsqueeze(-2)
                    probs = encoder(images, responses, rm)
                elif text_model == 'bert':
                    responses, labels = responses.to(device), labels.to(device)
                    rm = (responses != 0).int()
                    probs = encoder(images, responses, rm)

                loss = criterion(probs, labels)
                sum_loss_val += loss.item()
                for prob, label in zip(probs, labels):
                    p, l = prob.item(), label.item()
                    if ((p >= 0.5) and (l == 1.0)) or ((p < 0.5) and (l == 0.0)):
                       val_correct_count += 1

        val_accuracy = val_correct_count / len(val)

        print(str(datetime.now()).split('.')[0],
            "Epoch: %d/%d" %(epoch, epochs),
            "TrainLoss: %.3f" %(sum_loss_training / (len(train) / batch_size)),
            "TrainAccuracy: %.3f" %(training_accuracy),
            "ValLoss: %.3f" %(sum_loss_val / (len(val) / batch_size)),
            "ValAccuracy: %.3f" %(val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            if val_accuracy > 0.67:
                save_dir = 'models/' + task
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + '/' + text_model + '_' + image_model + '_' \
                            + datetime.now().strftime('%Y%m%d%H%M') + '_' + str(val_accuracy)[:5] + '.pt'
                torch.save(encoder.state_dict(), save_path)
                print("New best found and saved.")

        if val_accuracy + 0.01 < best_val_accuracy:
            print(str(datetime.now()).split('.')[0], "Training and validation epochs finished.")
            break


if text_model == 'bert':
    data_path = '../../data/' + task + '_bert_dataset.pkl'
else:
    data_path = '../../data/' + task + '_dataset.pkl'
with open(data_path, 'rb') as f:
    params = pickle.load(f)
    train = params['train']
    val = params['val']

batch_size = 64
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)

if text_model != 'bert':
    id_to_vec = train.id_to_vec
    emb_size = train.emb_dim
    vocab_size = len(train.word_to_id)

config = get_config('config/' + text_model + '_config.json')
if text_model == 'lstm':
    from model.image_encoder import ImageLstmEncoder
    encoder = ImageLstmEncoder(image_model, id_to_vec, emb_size, vocab_size, config)
elif text_model == 'transformer':
    from model.image_encoder import ImageTransformerEncoder
    encoder = ImageTransformerEncoder(image_model, id_to_vec, emb_size, vocab_size, config, device) 
elif text_model == 'bert':
    from model.image_encoder import ImageBertEncoder
    encoder = ImageBertEncoder(config)
encoder.to(device)

train_model(learning_rate=0.0001, l2_penalty=0.0001, epochs=50)
