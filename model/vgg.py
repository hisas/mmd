import torch
import torch.nn as nn
from torchvision import models


class VggEncoder(nn.Module):
    def __init__(self, embed_size, gaze=False):
        super(VggEncoder, self).__init__()

        self.gaze = gaze
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, embed_size)
        self.model = model
        if gaze:
            self.fc = nn.Linear(embed_size + 4, embed_size)

    def forward(self, image, gaze=None):
        image_feature = self.model(image)
        if self.gaze:
            image_feature = self.fc(torch.cat((image_feature, gaze), 1))

        return image_feature
