import torch
import torch.nn as nn
from torchvision import models

class VggEncoder(nn.Module):
    def __init__(self, embed_size):
        super(VggEncoder, self).__init__()

        model = models.vgg16(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        self.model = model
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, image):
        with torch.no_grad():
            image_feature = self.model(image)
        image_feature = self.fc(image_feature)
        l2_norm = image_feature.norm(p=2, dim=1, keepdim=True).detach()
        image_feature = image_feature.div(l2_norm)

        return image_feature
