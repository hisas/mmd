from torch import nn
from transformers import BertModel

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        model = BertModel.from_pretrained('bert-base-japanese-whole-word-masking')
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.in_features = model.pooler.dense.out_features
        self.fc = nn.Linear(self.in_features, config.hidden_size)

    def forward(self, input_ids, mask):
        encoded_layers, _ = self.model(input_ids, mask)

        vec_0 = encoded_layers[:, 0, :]
        vec_0 = vec_0.view(-1, 768)
        out = self.fc(vec_0)

        return out
