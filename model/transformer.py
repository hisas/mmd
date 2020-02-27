import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device, max_seq_len=40):
        super().__init__()

        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model).to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        ret = math.sqrt(self.d_model)*x + self.pe

        return ret

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, d_model, dropout):
        super().__init__()

        self.h = nhead
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask, dropout):
        weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)
        normlized_weights = F.softmax(weights, dim=-1)
        normlized_weights = dropout(normlized_weights)
        output = torch.matmul(normlized_weights, v)

        return output

    def forward(self, q, k, v, mask):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout, d_ff=1024):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, nhead, d_model, dropout):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(nhead, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_normlized = self.norm_1(x)
        output = self.attn(x_normlized, x_normlized, x_normlized, mask)
        x2 = x + self.dropout_1(output)
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output

class Aggregator(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()

        self.linear = nn.Linear(d_model, output_dim)
        init.normal_(self.linear.weight, std=0.02)
        init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]
        out = self.linear(x0)

        return out

class TransformerEncoder(nn.Module):
    def __init__(self, id_to_vec, emb_size, vocab_size, config, device):
        super(TransformerEncoder, self).__init__()

        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        hidden_size, dropout, nhead, d_model = config.hidden_size, config.dropout, config.nhead, config.d_model 
        self.pos_encoder = PositionalEncoder(d_model, device)
        self.transformer_1 = TransformerBlock(nhead, d_model, dropout)
        self.transformer_2 = TransformerBlock(nhead, d_model, dropout)
        self.aggregator = Aggregator(d_model, hidden_size)
        self.init_weights(id_to_vec)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    def init_weights(self, id_to_vec):
        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_size)
        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)

        self.transformer_1.apply(self.weights_init)
        self.transformer_2.apply(self.weights_init)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_1(x, mask)
        x = self.transformer_2(x, mask)
        out = self.aggregator(x)

        return out
