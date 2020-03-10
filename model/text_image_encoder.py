import pathlib
import sys

import torch
from torch import nn
from torch.nn import init
from block import fusions


class TextImageLstmEncoder(nn.Module):
    def __init__(self, image_model, fusion_method, id_to_vec, emb_size, vocab_size, config):
        super(TextImageLstmEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif fusion_method == 'mcb':
            self.fusion = fusions.MCB([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'mlb':
            self.fusion = fusions.MLB([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'mutan':
            self.fusion = fusions.Mutan([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'block':
            self.fusion = fusions.Block([self.hidden_size, self.hidden_size], self.hidden_size)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_encoder = VggEncoder(self.hidden_size)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_encoder = ResNetEncoder(self.hidden_size)

        from model.lstm import LstmEncoder
        self.context_encoder = LstmEncoder(id_to_vec, emb_size, vocab_size, config)
        self.response_encoder = LstmEncoder(id_to_vec, emb_size, vocab_size, config)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, contexts_len, csi, images, responses, responses_len, rsi):
        contexts_last_hidden = self.context_encoder(contexts, contexts_len)
        images_feature = self.image_encoder(images)
        responses_last_hidden = self.response_encoder(responses, responses_len)
        sorted_c = torch.Tensor(len(contexts), self.hidden_size).to(contexts.device)
        sorted_r = torch.Tensor(len(responses), self.hidden_size).to(responses.device)
        for i, v in enumerate(csi):
            sorted_c[v] = contexts_last_hidden[i]
        for i, v in enumerate(rsi):
            sorted_r[v] = responses_last_hidden[i]

        if self.fusion_method == 'concat':
            contexts_images = self.fc(torch.cat((sorted_c, images_feature), dim=1))
        elif self.fusion_method == 'sum':
            contexts_images = sorted_c + images_feature
        elif self.fusion_method == 'product':
            contexts_images = sorted_c * images_feature
        elif self.fusion_method in ['mcb', 'mlb', 'mutan', 'block']:
            contexts_images = self.fusion([sorted_c, images_feature])

        contexts_images = contexts_images.mm(self.M)
        contexts_images = contexts_images.view(-1, 1, self.hidden_size)
        responses = sorted_r.view(-1, self.hidden_size, 1)
        score = torch.bmm(contexts_images, responses)
        prob = torch.sigmoid(score).view(-1, 1)
        
        return prob


class TextImageTransformerEncoder(nn.Module):
    def __init__(self, image_model, fusion_method, id_to_vec, emb_size, vocab_size, config, device='cuda:0'):
        super(TextImageTransformerEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif fusion_method == 'mcb':
            self.fusion = fusions.MCB([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'mlb':
            self.fusion = fusions.MLB([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'mutan':
            self.fusion = fusions.Mutan([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'block':
            self.fusion = fusions.Block([self.hidden_size, self.hidden_size], self.hidden_size)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_encoder = VggEncoder(self.hidden_size)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_encoder = ResNetEncoder(self.hidden_size)

        from model.transformer import TransformerEncoder
        self.context_encoder = TransformerEncoder(id_to_vec, emb_size, vocab_size, config, device)
        self.response_encoder = TransformerEncoder(id_to_vec, emb_size, vocab_size, config, device)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, images, responses, rm):
        contexts_first = self.context_encoder(contexts, cm)
        images_feature = self.image_encoder(images)
        responses_first = self.response_encoder(responses, rm)

        if self.fusion_method == 'concat':
            contexts_images = self.fc(torch.cat((contexts_first, images_feature), dim=1))
        elif self.fusion_method == 'sum':
            contexts_images = contexts_first + images_feature
        elif self.fusion_method == 'product':
            contexts_images = contexts_first * images_feature
        elif self.fusion_method in ['mcb', 'mlb', 'mutan', 'block']:
            contexts_images = self.fusion([contexts_first, images_feature])
        
        contexts_images = contexts_images.mm(self.M)
        contexts_images = contexts_images.view(-1, 1, self.hidden_size)
        responses = responses_first.view(-1, self.hidden_size, 1)
        score = torch.bmm(contexts_images, responses)
        prob = torch.sigmoid(score).view(-1, 1)

        return prob

class TextImageBertEncoder(nn.Module):
    def __init__(self, image_model, fusion_method, config):
        super(TextImageBertEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif fusion_method == 'mcb':
            self.fusion = fusions.MCB([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'mlb':
            self.fusion = fusions.MLB([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'mutan':
            self.fusion = fusions.Mutan([self.hidden_size, self.hidden_size], self.hidden_size)
        elif fusion_method == 'block':
            self.fusion = fusions.Block([self.hidden_size, self.hidden_size], self.hidden_size)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_encoder = VggEncoder(self.hidden_size)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_encoder = ResNetEncoder(self.hidden_size)

        from model.bert import BertEncoder
        self.text_encoder = BertEncoder(config)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, images, responses, rm):
        contexts_first = self.text_encoder(contexts, cm)
        images_feature = self.image_encoder(images)
        responses_first = self.text_encoder(responses, rm)

        if self.fusion_method == 'concat':
            contexts_images = self.fc(torch.cat((contexts_first, images_feature), dim=1))
        elif self.fusion_method == 'sum':
            contexts_images = contexts_first + images_feature
        elif self.fusion_method == 'product':
            contexts_images = contexts_first * images_feature
        elif self.fusion_method in ['mcb', 'mlb', 'mutan', 'block']:
            contexts_images = self.fusion([contexts_first, images_feature])
        
        contexts_images = contexts_images.mm(self.M)
        contexts_images = contexts_images.view(-1, 1, self.hidden_size)
        responses = responses_first.view(-1, self.hidden_size, 1)
        score = torch.bmm(contexts_images, responses)
        prob = torch.sigmoid(score).view(-1, 1)

        return prob
