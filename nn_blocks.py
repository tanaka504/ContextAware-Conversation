import torch
import torch.nn as nn
import torch.nn.functional as F


class DAEncoder(nn.Module):
    def __init__(self, da_input_size, da_embed_size,da_hidden):
        super(DAEncoder, self).__init__()
        self.hidden_size = da_hidden
        self.xe = nn.Embedding(da_input_size, da_embed_size)
        self.eh = nn.Linear(da_embed_size, da_hidden)

    def forward(self, DA):
        embedding = F.tanh(self.eh(self.xe(DA)))
        return embedding

    def initHidden(self, device):
        return torch.zeros(self.hidden_size).to(device)


class DAContextEncoder(nn.Module):
    def __init__(self, da_hidden):
        super(DAContextEncoder, self).__init__()
        self.hidden_size = da_hidden
        self.hh = nn.GRU(da_hidden, da_hidden, batch_first=True)

    def forward(self, input_hidden, prev_hidden):
        output = input_hidden.view(1, 1, -1)
        output, hidden = self.hh(output, prev_hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size).to(device)


class DADecoder(nn.Module):
    def __init__(self, da_input_size, da_embed_size, da_hidden):
        super(DADecoder, self).__init__()
        self.he = nn.Linear(da_hidden, da_embed_size)
        self.ey = nn.Linear(da_embed_size, da_input_size)

    def forward(self, hidden):
        pred = self.ey(F.tanh(self.he(hidden)))
        return pred


class UtteranceEncoder(nn.Module):
    def __init__(self, utt_input_size, embed_size, utterance_hidden):
        super(UtteranceEncoder, self).__init__()
        self.hidden_size = utterance_hidden
        self.xe = nn.Embedding(utt_input_size, embed_size)
        self.eh = nn.Linear(embed_size, utterance_hidden)
        self.hh = nn.GRU(utterance_hidden, utterance_hidden, num_layers=1, batch_first=True)

    def forward(self, X, hidden):
        embedding = F.tanh(self.eh(self.xe(X))).view(1, 1, -1)
        output = embedding
        output, hidden = self.hh(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size).to(device)


class UtteranceContextEncoder(nn.Module):
    def __init__(self, utterance_hidden_size):
        super(UtteranceContextEncoder, self).__init__()
        self.hidden_size = utterance_hidden_size
        self.hh = nn.GRU(utterance_hidden_size, utterance_hidden_size, batch_first=True)

    def forward(self, input_hidden, prev_hidden):
        output = input_hidden.view(1, 1, -1)
        output, hidden = self.hh(output, prev_hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size).to(device)




