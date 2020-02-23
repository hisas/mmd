import os
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../..')
import argparse
import json
from attrdict import AttrDict
import numpy as np
import random
import pickle
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from mmd_dataset import MmdDataset
from helper.image_helper import *

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
parser.add_argument('--synthesis_method')
args = parser.parse_args()
task = args.task
text_model = args.text_model
image_model = args.image_model
synthesis_method = args.synthesis_method


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

    for epoch in range(epochs):
        sum_loss_training = 0.0
        training_correct_count = 0
        encoder.train()

        for contexts, contexts_len, images_path, _, responses, responses_len, labels in tqdm(train_loader):
            encoder.zero_grad()

            images = fetch_images(images_path, 'train')
            images = images.to(device)

            if text_model == 'lstm':
                contexts_len, csi = contexts_len.sort(descending=True)
                sorted_contexts = contexts[csi]
                responses_len, rsi = responses_len.sort(descending=True)
                sorted_responses = responses[rsi]
                sorted_contexts = sorted_contexts.to(device)
                sorted_responses = sorted_responses.to(device)
                labels = labels.to(device)
                probs = encoder(sorted_contexts, contexts_len, csi, images, sorted_responses, responses_len, rsi)
            elif text_model == 'transformer':
                contexts, responses, labels = contexts.to(device), responses.to(device), labels.to(device)
                cm, rm = (contexts != 0).unsqueeze(-2), (responses != 0).unsqueeze(-2)
                probs = encoder(contexts, cm, images, responses, rm)

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

        for contexts, contexts_len, images_path, _, responses, responses_len, labels in tqdm(val_loader):
            with torch.no_grad():
                images = fetch_images(images_path, 'test')
                images = images.to(device)

                if text_model == 'lstm':
                    contexts_len, csi = contexts_len.sort(descending=True)
                    sorted_contexts = contexts[csi]
                    responses_len, rsi = responses_len.sort(descending=True)
                    sorted_responses = responses[rsi]
                    sorted_contexts = sorted_contexts.to(device)
                    sorted_responses = sorted_responses.to(device)
                    labels = labels.to(device)
                    probs = encoder(sorted_contexts, contexts_len, csi, images, sorted_responses, responses_len, rsi)
                elif text_model == 'transformer':
                    contexts, responses, labels = contexts.to(device), responses.to(device), labels.to(device)
                    cm, rm = (contexts != 0).unsqueeze(-2), (responses != 0).unsqueeze(-2)
                    probs = encoder(contexts, cm, images, responses, rm)

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

            if val_accuracy > 0.7:                
                save_dir = 'models/' + task
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + '/' + text_model + '_' + image_model + '_' + synthesis_method + '_' \
                            + datetime.now().strftime('%Y%m%d%H%M') + '_' + str(val_accuracy)[:5] + '.pt'
                torch.save(encoder.state_dict(), save_path)
                print("New best found and saved.")

        if val_accuracy + 0.01 < best_val_accuracy:
            print(str(datetime.now()).split('.')[0], "Training and validation epochs finished.")
            break


with open('../../data/' + task + '_dataset.pkl', 'rb') as f:
    params = pickle.load(f)
    train = params['train']
    val = params['val']

batch_size = 64
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4)

id_to_vec = train.id_to_vec
emb_size = train.emb_dim
vocab_size = len(train.word_to_id)

if text_model == 'lstm':
    config = get_config('config/lstm_config.json')
    from model.text_image_encoder import TextImageLstmEncoder
    encoder = TextImageLstmEncoder(image_model, synthesis_method, id_to_vec, emb_size, vocab_size, config)
elif text_model == 'transformer':
    config = get_config('config/transformer_config.json')
    from model.text_image_encoder import TextImageTransformerEncoder
    encoder = TextImageTransformerEncoder(image_model, synthesis_method, id_to_vec, emb_size, vocab_size, config, device) 
encoder.to(device)

train_model(learning_rate=0.0001, l2_penalty=0.0001, epochs=50)
