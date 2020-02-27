import torch
from torch import nn


class LstmEncoder(nn.Module):
    def __init__(self, id_to_vec, emb_size, vocab_size, config):
        super(LstmEncoder, self).__init__()

        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, config.hidden_size, batch_first=True, dropout=config.dropout)
        self.init_weights(id_to_vec)

    def forward(self, x, l):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]

        return last_hidden

    def init_weights(self, id_to_vec):
        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_size)

        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec

        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)
