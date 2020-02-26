import torch
from torch import nn
from torch.nn import init

class TextLstmEncoder(nn.Module):
    def __init__(self, id_to_vec, emb_size, vocab_size, config):
        super(TextLstmEncoder, self).__init__()

        from model.lstm import LstmEncoder
        self.encoder = LstmEncoder(id_to_vec, emb_size, vocab_size, config)
        self.hidden_size = config.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, contexts_len, csi, responses, responses_len, rsi):
        contexts_last_hidden = self.encoder(contexts, contexts_len)
        responses_last_hidden = self.encoder(responses, responses_len)
        clh = torch.Tensor(len(contexts), self.hidden_size).to(contexts.device)
        rlh = torch.Tensor(len(responses), self.hidden_size).to(responses.device)
        for i, v in enumerate(csi):
            clh[v] = contexts_last_hidden[i]
        for i, v in enumerate(rsi):
            rlh[v] = responses_last_hidden[i]

        contexts = clh.mm(self.M)
        contexts = contexts.view(-1, 1, self.hidden_size)
        responses = rlh.view(-1, self.hidden_size, 1)
        score = torch.bmm(contexts, responses)
        probs = torch.sigmoid(score.view(-1, 1))

        return probs

class TextTransformerEncoder(nn.Module):
    def __init__(self, id_to_vec, emb_size, vocab_size, config, device='cuda:0'):
        super(TextTransformerEncoder, self).__init__()

        from model.transformer import TransformerEncoder
        self.encoder = TransformerEncoder(id_to_vec, emb_size, vocab_size, config, device)
        self.hidden_size = config.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, responses, rm):
        contexts_first = self.encoder(contexts, cm)
        responses_first = self.encoder(responses, rm)
        contexts = contexts_first.mm(self.M)
        contexts = contexts.view(-1, 1, self.hidden_size)
        responses = responses_first.view(-1, self.hidden_size, 1)
        score = torch.bmm(contexts, responses)
        probs = torch.sigmoid(score.view(-1, 1))

        return probs

class TextBertEncoder(nn.Module):
    def __init__(self, config):
        super(TextBertEncoder, self).__init__()

        from model.bert import BertEncoder
        self.encoder = BertEncoder(config)
        self.hidden_size = config.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, contexts, cm, responses, rm):
        contexts_first = self.encoder(contexts, cm)
        responses_first = self.encoder(responses, rm)
        contexts = contexts_first.mm(self.M)
        contexts = contexts.view(-1, 1, self.hidden_size)
        responses = responses_first.view(-1, self.hidden_size, 1)
        score = torch.bmm(contexts, responses)
        probs = torch.sigmoid(score.view(-1, 1))

        return probs
