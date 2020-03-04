import pathlib
import sys

import torch
from torch import nn
from torch.nn import init

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/..')

from helper.compact_bilinear_pooling import CompactBilinearPooling
from helper.fusion import MLBFusion, MutanFusion


class TextImageLstmEncoder(nn.Module):
    def __init__(self, image_model, fusion_method, id_to_vec, emb_size, vocab_size, config):
        super(TextImageLstmEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif fusion_method == 'mcb':
            self.fusion = CompactBilinearPooling(self.hidden_size, self.hidden_size, self.hidden_size)
        elif fusion_method == 'mlb':
            self.fusion = MLBFusion({'dim_h': self.hidden_size, 'dropout_v': 0.5, 'dropout_q': 0.5})
        elif fusion_method == 'mutan':
            self.fusion = MutanFusion({'dim_hv': self.hidden_size, 'dim_hq': self.hidden_size, 'dim_mm': self.hidden_size, \
                                       'R': 5, 'dropout_hv': 0, 'dropout_hq': 0}, visual_embedding=False, question_embedding=False)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_encoder = VggEncoder(self.hidden_size)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_encoder = ResNetEncoder(self.hidden_size)
        elif image_model == 'efficientnet':
            from model.efficientnet import EfficientNetEncoder
            self.image_encoder = EfficientNetEncoder(self.hidden_size)

        from model.lstm import LstmEncoder
        self.text_encoder = LstmEncoder(id_to_vec, emb_size, vocab_size, config)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, contexts_len, csi, images, responses, responses_len, rsi):
        contexts_last_hidden = self.text_encoder(contexts, contexts_len)
        images_feature = self.image_encoder(images)
        responses_last_hidden = self.text_encoder(responses, responses_len)
        sorted_c = torch.Tensor(len(contexts), self.hidden_size).to(contexts.device)
        sorted_r = torch.Tensor(len(responses), self.hidden_size).to(responses.device)
        for i, v in enumerate(csi):
            sorted_c[v] = contexts_last_hidden[i]
        for i, v in enumerate(rsi):
            sorted_r[v] = responses_last_hidden[i]

        if self.fusion_method == 'late':
            contexts = sorted_c.mm(self.M)
            contexts = contexts.view(-1, 1, self.hidden_size)
            images = images_feature.view(-1, 1, self.hidden_size)
            responses = sorted_r.view(-1, self.hidden_size, 1)
            score_1, score_2 = torch.bmm(contexts, responses), torch.bmm(images, responses)
            probs_1, probs_2 = torch.sigmoid(score_1), torch.sigmoid(score_2)
            prob = torch.bmm(probs_1, probs_2).view(-1, 1)
        elif self.fusion_method == 'concat':
            contexts_images = self.fc(torch.cat((sorted_c, images_feature), dim=1))
            responses_images = self.fc(torch.cat((sorted_r, images_feature), dim=1))
        elif self.fusion_method == 'sum':
            contexts_images = sorted_c + images_feature
            responses_images = sorted_r + images_feature
        elif self.fusion_method == 'product':
            contexts_images = sorted_c * images_feature
            responses_images = sorted_r * images_feature
        elif self.fusion_method in ['mcb', 'mlb', 'mutan']:
            contexts_images = self.fusion(sorted_c, images_feature)
            responses_images = self.fusion(sorted_r, images_feature)

        if self.fusion_method != 'late':
            contexts_images = contexts_images.mm(self.M)
            contexts_images = contexts_images.view(-1, 1, self.hidden_size)
            responses_images = responses_images.view(-1, self.hidden_size, 1)
            score = torch.bmm(contexts_images, responses_images)
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
            self.fusion = CompactBilinearPooling(self.hidden_size, self.hidden_size, self.hidden_size)
        elif fusion_method == 'mlb':
            self.fusion = MLBFusion({'dim_h': self.hidden_size, 'dropout_v': 0.5, 'dropout_q': 0.5})
        elif fusion_method == 'mutan':
            self.fusion = MutanFusion({'dim_hv': self.hidden_size, 'dim_hq': self.hidden_size, 'dim_mm': self.hidden_size, \
                                       'R': 5, 'dropout_hv': 0, 'dropout_hq': 0}, visual_embedding=False, question_embedding=False)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_encoder = VggEncoder(self.hidden_size)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_encoder = ResNetEncoder(self.hidden_size)
        elif image_model == 'efficientnet':
            from model.efficientnet import EfficientNetEncoder
            self.image_encoder = EfficientNetEncoder(self.hidden_size)

        from model.transformer import TransformerEncoder
        self.text_encoder = TransformerEncoder(id_to_vec, emb_size, vocab_size, config, device)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, images, responses, rm):
        contexts_first = self.text_encoder(contexts, cm)
        images_feature = self.image_encoder(images)
        responses_first = self.text_encoder(responses, rm)

        if self.fusion_method == 'late':
            contexts = contexts_first.mm(self.M)
            contexts = contexts.view(-1, 1, self.hidden_size)
            images = images_feature.view(-1, 1, self.hidden_size)
            responses = responses_first.view(-1, self.hidden_size, 1)
            score_1, score_2 = torch.bmm(contexts, responses), torch.bmm(images, responses)
            probs_1, probs_2 = torch.sigmoid(score_1), torch.sigmoid(score_2)
            prob = torch.bmm(probs_1, probs_2).view(-1, 1)
        elif self.fusion_method == 'concat':
            contexts_images = self.fc(torch.cat((contexts_first, images_feature), dim=1))
            responses_images = self.fc(torch.cat((responses_first, images_feature), dim=1))
        elif self.fusion_method == 'sum':
            contexts_images = contexts_first + images_feature
            responses_images = responses_first + images_feature
        elif self.fusion_method == 'product':
            contexts_images = contexts_first * images_feature
            responses_images = responses_first * images_feature
        elif self.fusion_method in ['mcb', 'mlb', 'mutan']:
            contexts_images = self.fusion(contexts_first, images_feature)
            responses_images = self.fusion(responses_first, images_feature)
        
        if self.fusion_method != 'late':
            contexts_images = contexts_images.mm(self.M)
            contexts_images = contexts_images.view(-1, 1, self.hidden_size)
            responses_images = responses_images.view(-1, self.hidden_size, 1)
            score = torch.bmm(contexts_images, responses_images)
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
            self.fusion = CompactBilinearPooling(self.hidden_size, self.hidden_size, self.hidden_size)
        elif fusion_method == 'mlb':
            self.fusion = MLBFusion({'dim_h': self.hidden_size, 'dropout_v': 0.5, 'dropout_q': 0.5})
        elif fusion_method == 'mutan':
            self.fusion = MutanFusion({'dim_hv': self.hidden_size, 'dim_hq': self.hidden_size, 'dim_mm': self.hidden_size, \
                                       'R': 5, 'dropout_hv': 0, 'dropout_hq': 0}, visual_embedding=False, question_embedding=False)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_encoder = VggEncoder(self.hidden_size)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_encoder = ResNetEncoder(self.hidden_size)
        elif image_model == 'efficientnet':
            from model.efficientnet import EfficientNetEncoder
            self.image_encoder = EfficientNetEncoder(self.hidden_size)

        from model.bert import BertEncoder
        self.text_encoder = BertEncoder(config)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, images, responses, rm):
        contexts_first = self.text_encoder(contexts, cm)
        images_feature = self.image_encoder(images)
        responses_first = self.text_encoder(responses, rm)

        if self.fusion_method == 'late':
            contexts = contexts_first.mm(self.M)
            contexts = contexts.view(-1, 1, self.hidden_size)
            images = images_feature.view(-1, 1, self.hidden_size)
            responses = responses_first.view(-1, self.hidden_size, 1)
            score_1, score_2 = torch.bmm(contexts, responses), torch.bmm(images, responses)
            probs_1, probs_2 = torch.sigmoid(score_1), torch.sigmoid(score_2)
            prob = torch.bmm(probs_1, probs_2).view(-1, 1)
        elif self.fusion_method == 'concat':
            contexts_images = self.fc(torch.cat((contexts_first, images_feature), dim=1))
            responses_images = self.fc(torch.cat((responses_first, images_feature), dim=1))
        elif self.fusion_method == 'sum':
            contexts_images = contexts_first + images_feature
            responses_images = responses_first + images_feature
        elif self.fusion_method == 'product':
            contexts_images = contexts_first * images_feature
            responses_images = responses_first * images_feature
        elif self.fusion_method in ['mcb', 'mlb', 'mutan']:
            contexts_images = self.fusion(contexts_first, images_feature)
            responses_images = self.fusion(responses_first, images_feature)
        
        if self.fusion_method != 'late':
            contexts_images = contexts_images.mm(self.M)
            contexts_images = contexts_images.view(-1, 1, self.hidden_size)
            responses_images = responses_images.view(-1, self.hidden_size, 1)
            score = torch.bmm(contexts_images, responses_images)
            prob = torch.sigmoid(score).view(-1, 1)

        return prob

