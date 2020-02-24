import torch
from torch import nn
from torch.nn import init

class ImageLstmEncoder(nn.Module):
    def __init__(self, image_model, id_to_vec, emb_size, vocab_size, config):
        super(ImageLstmEncoder, self).__init__()

        self.hidden_size = config.hidden_size
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

    def forward(self, images, responses, responses_len, rsi):
        images_feature = self.image_encoder(images)
        responses_last_hidden = self.text_encoder(responses, responses_len)
        sorted_r = torch.Tensor(len(responses), self.hidden_size).to(responses.device)
        for i, v in enumerate(rsi):
            sorted_r[v] = responses_last_hidden[i]

        images = images_feature.mm(self.M).view(-1, 1, self.hidden_size)
        responses = sorted_r.view(-1, self.hidden_size, 1)
        score = torch.bmm(images, responses)
        prob = torch.sigmoid(score).view(-1, 1)
        
        return prob


class ImageTransformerEncoder(nn.Module):
    def __init__(self, image_model, id_to_vec, emb_size, vocab_size, config, device='cuda:0'):
        super(ImageTransformerEncoder, self).__init__()

        self.hidden_size = config.hidden_size
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

    def forward(self, images, responses, rm):
        images_feature = self.image_encoder(images)
        responses_first = self.text_encoder(responses, rm)

        images = images_feature.mm(self.M).view(-1, 1, self.hidden_size)
        responses = responses_first.view(-1, self.hidden_size, 1)
        score = torch.bmm(images, responses)
        prob = torch.sigmoid(score).view(-1, 1)

        return prob
