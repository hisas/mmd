import pathlib
import sys

import torch
from torch import nn
from torch.nn import init

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/..')

from helper.compact_bilinear_pooling import CompactBilinearPooling


class TextImageGazeLstmEncoder(nn.Module):
    def __init__(self, image_model, joint_method, id_to_vec, emb_size, vocab_size, config):
        super(TextImageGazeLstmEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.joint_method = joint_method
        if joint_method == 'concat':
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif joint_method == 'mcb':
            self.mcb = CompactBilinearPooling(self.hidden_size, self.hidden_size, self.hidden_size)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_gaze_encoder = VggEncoder(self.hidden_size, gaze=True)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_gaze_encoder = ResNetEncoder(self.hidden_size)
        elif image_model == 'efficientnet':
            from model.efficientnet import EfficientNetEncoder
            self.image_gaze_encoder = EfficientNetEncoder(self.hidden_size)

        from model.lstm import LstmEncoder
        self.text_encoder = LstmEncoder(id_to_vec, emb_size, vocab_size, config)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, contexts_len, csi, images, gazes, responses, responses_len, rsi):
        contexts_last_hidden = self.text_encoder(contexts, contexts_len)
        images_gazes_feature = self.image_gaze_encoder(images, gazes)
        responses_last_hidden = self.text_encoder(responses, responses_len)
        sorted_c = torch.Tensor(len(contexts), self.hidden_size).to(contexts.device)
        sorted_r = torch.Tensor(len(responses), self.hidden_size).to(responses.device)
        for i, v in enumerate(csi):
            sorted_c[v] = contexts_last_hidden[i]
        for i, v in enumerate(rsi):
            sorted_r[v] = responses_last_hidden[i]

        if self.joint_method == 'late':
            contexts = sorted_c.mm(self.M)
            contexts = contexts.view(-1, 1, self.hidden_size)
            images_gazes = images_gazes_feature.view(-1, 1, self.hidden_size)
            responses = sorted_r.view(-1, self.hidden_size, 1)
            score_1, score_2 = torch.bmm(contexts, responses), torch.bmm(images_gazes, responses)
            probs_1, probs_2 = torch.sigmoid(score_1), torch.sigmoid(score_2)
            prob = torch.bmm(probs_1, probs_2).view(-1, 1)
        elif self.joint_method == 'concat':
            contexts_images = self.fc(torch.cat((sorted_c, images_gazes_feature), dim=1))
            responses_images = self.fc(torch.cat((sorted_r, images_gazes_feature), dim=1))
        elif self.joint_method == 'sum':
            contexts_images = sorted_c + images_gazes_feature
            responses_images = sorted_r + images_gazes_feature
        elif self.joint_method == 'product':
            contexts_images = sorted_c * images_gazes_feature
            responses_images = sorted_r * images_gazes_feature
        elif self.joint_method == 'mcb':
            contexts_images = self.mcb(sorted_c, images_gazes_feature)
            responses_images = self.mcb(sorted_r, images_gazes_feature)
        
        if self.joint_method != 'late':
            contexts_images = contexts_images.mm(self.M)
            contexts_images = contexts_images.view(-1, 1, self.hidden_size)
            responses_images = responses_images.view(-1, self.hidden_size, 1)
            score = torch.bmm(contexts_images, responses_images)
            prob = torch.sigmoid(score).view(-1, 1)
        
        return prob


class TextImageGazeTransformerEncoder(nn.Module):
    def __init__(self, image_model, joint_method, id_to_vec, emb_size, vocab_size, config, device='cuda:0'):
        super(TextImageGazeTransformerEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.joint_method = joint_method
        if joint_method == 'concat':
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif joint_method == 'mcb':
            self.mcb = CompactBilinearPooling(self.hidden_size, self.hidden_size, self.hidden_size)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_gaze_encoder = VggEncoder(self.hidden_size, gaze=True)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_gaze_encoder = ResNetEncoder(self.hidden_size)
        elif image_model == 'efficientnet':
            from model.efficientnet import EfficientNetEncoder
            self.image_gaze_encoder = EfficientNetEncoder(self.hidden_size)

        from model.transformer import TransformerEncoder
        self.text_encoder = TransformerEncoder(id_to_vec, emb_size, vocab_size, config, device)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, images, gazes, responses, rm):
        contexts_first = self.text_encoder(contexts, cm)
        images_gazes_feature = self.image_gaze_encoder(images, gazes)
        responses_first = self.text_encoder(responses, rm)

        if self.joint_method == 'late':
            contexts = contexts_first.mm(self.M)
            contexts = contexts.view(-1, 1, self.hidden_size)
            images = images_gazes_feature.view(-1, 1, self.hidden_size)
            responses = responses_first.view(-1, self.hidden_size, 1)
            score_1, score_2 = torch.bmm(contexts, responses), torch.bmm(images, responses)
            probs_1, probs_2 = torch.sigmoid(score_1), torch.sigmoid(score_2)
            prob = torch.bmm(probs_1, probs_2).view(-1, 1)
        elif self.joint_method == 'concat':
            contexts_images = self.fc(torch.cat((contexts_first, images_gazes_feature), dim=1))
            responses_images = self.fc(torch.cat((responses_first, images_gazes_feature), dim=1))
        elif self.joint_method == 'sum':
            contexts_images = contexts_first + images_gazes_feature
            responses_images = responses_first + images_gazes_feature
        elif self.joint_method == 'product':
            contexts_images = contexts_first * images_gazes_feature
            responses_images = responses_first * images_gazes_feature
        elif self.joint_method == 'mcb':
            contexts_images = self.mcb(contexts_first, images_gazes_feature)
            responses_images = self.mcb(responses_first, images_gazes_feature)
        
        if self.joint_method != 'late':
            contexts_images = contexts_images.mm(self.M)
            contexts_images = contexts_images.view(-1, 1, self.hidden_size)
            responses_images = responses_images.view(-1, self.hidden_size, 1)
            score = torch.bmm(contexts_images, responses_images)
            prob = torch.sigmoid(score).view(-1, 1)

        return prob

class TextImageGazeBertEncoder(nn.Module):
    def __init__(self, image_model, joint_method, config):
        super(TextImageGazeBertEncoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.joint_method = joint_method
        if joint_method == 'concat':
            self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        elif joint_method == 'mcb':
            self.mcb = CompactBilinearPooling(self.hidden_size, self.hidden_size, self.hidden_size)

        if image_model == 'vgg':
            from model.vgg import VggEncoder
            self.image_gaze_encoder = VggEncoder(self.hidden_size, gaze=True)
        elif image_model == 'resnet':
            from model.resnet import ResNetEncoder
            self.image_gaze_encoder = ResNetEncoder(self.hidden_size)
        elif image_model == 'efficientnet':
            from model.efficientnet import EfficientNetEncoder
            self.image_gaze_encoder = EfficientNetEncoder(self.hidden_size)

        from model.bert import BertEncoder
        self.text_encoder = BertEncoder(config)
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, images, gazes, responses, rm):
        contexts_first = self.text_encoder(contexts, cm)
        images_gazes_feature = self.image_gaze_encoder(images, gazes)
        responses_first = self.text_encoder(responses, rm)

        if self.joint_method == 'late':
            contexts = contexts_first.mm(self.M)
            contexts = contexts.view(-1, 1, self.hidden_size)
            images = images_gazes_feature.view(-1, 1, self.hidden_size)
            responses = responses_first.view(-1, self.hidden_size, 1)
            score_1, score_2 = torch.bmm(contexts, responses), torch.bmm(images, responses)
            probs_1, probs_2 = torch.sigmoid(score_1), torch.sigmoid(score_2)
            prob = torch.bmm(probs_1, probs_2).view(-1, 1)
        elif self.joint_method == 'concat':
            contexts_images = self.fc(torch.cat((contexts_first, images_gazes_feature), dim=1))
            responses_images = self.fc(torch.cat((responses_first, images_gazes_feature), dim=1))
        elif self.joint_method == 'sum':
            contexts_images = contexts_first + images_gazes_feature
            responses_images = responses_first + images_gazes_feature
        elif self.joint_method == 'product':
            contexts_images = contexts_first * images_gazes_feature
            responses_images = responses_first * images_gazes_feature
        elif self.joint_method == 'mcb':
            contexts_images = self.mcb(contexts_first, images_gazes_feature)
            responses_images = self.mcb(responses_first, images_gazes_feature)
        
        if self.joint_method != 'late':
            contexts_images = contexts_images.mm(self.M)
            contexts_images = contexts_images.view(-1, 1, self.hidden_size)
            responses_images = responses_images.view(-1, self.hidden_size, 1)
            score = torch.bmm(contexts_images, responses_images)
            prob = torch.sigmoid(score).view(-1, 1)

        return prob

