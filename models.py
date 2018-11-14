import torch
import torch.nn as nn
from nn_blocks import *
from torch import optim
import time


class DAestimator(nn.Module):
    def __init__(self):
        super(DAestimator, self).__init__()

    def forward(self, X_tensor, Y_tensor,
              encoder, decoder, context, context_hidden,
              criterion, last):
        loss = 0

        encoder_hidden = encoder(X_tensor)

        context_output, context_hidden = context(encoder_hidden, context_hidden)

        decoder_output = decoder(context_hidden)
        decoder_output = decoder_output.squeeze(1)

        loss += criterion(decoder_output, Y_tensor)

        if last:
            loss.backward()
            return loss.data[0], context_hidden
        else:
            return context_hidden

    def evaluate(self, X_tensor, Y_tensor,
                 encoder, decoder, context, context_hidden,
                 criterion):
        loss = 0
        encoder_hidden = encoder(X_tensor)
        context_output, context_hidden = context(encoder_hidden, context_hidden)
        decoder_output = decoder(context_hidden)
        decoder_output = decoder_output.squeeze(1)
        loss += criterion(decoder_output, Y_tensor)
        return loss.data[0], context_hidden
