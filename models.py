import torch
import torch.nn as nn
from nn_blocks import *
from torch import optim
import time
from train import * 


class DApredictModel(nn.Module):
    def __init__(self):
        super(DApredictModel, self).__init__()

    def forward(self, X_da, Y_da, X_utt, Y_utt, step_size,
                da_encoder, da_decoder, da_context, da_context_hidden,
                utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                criterion, last, config):
        loss = 0

        da_encoder_hidden = da_encoder(X_da) # (batch_size, 1, DA_HIDDEN)
        da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden) # (batch_size, 1, DA_HIDDEN)

        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(step_size, device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden) # (batch_size, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2) # (batch_size, 1, DEC_HIDDEN)
        elif config['use_uttcontext']:
            utt_encoder_hidden = utt_encoder.initHidden(step_size, device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)
            utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (batch_size, 1, DEC_HIDDEN)
        else:
            dec_hidden = da_context_output


        decoder_output = da_decoder(dec_hidden) # (batch_size, 1, DA_VOCAB)
        decoder_output = decoder_output.squeeze(1) # (batch_size, DA_VOCAB)
        Y_da = Y_da.squeeze()


        loss += criterion(decoder_output, Y_da)

        if last:
            loss.backward()
            return loss.item(), da_context_hidden, utt_context_hidden
        else:
            return da_context_hidden, utt_context_hidden

    def evaluate(self, X_da, Y_da, X_utt, Y_utt,
                 da_encoder, da_decoder, da_context, da_context_hidden,
                 utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                 criterion, config):
        loss = 0
        da_encoder_hidden = da_encoder(X_da)
        da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden)

        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(1, device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden) # (1, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2) # (1, 1, DEC_HIDDEN)
        elif config['use_uttcontext']:
            utt_encoder_hidden = utt_encoder.initHidden(1, device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
            utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (1, 1, DEC_HIDDEN)
        else:
            dec_hidden = da_context_output

        decoder_output = da_decoder(dec_hidden)
        decoder_output = decoder_output.squeeze(1)
        Y_da = Y_da.squeeze(0)

        loss += criterion(decoder_output, Y_da)

        return loss.item(), da_context_hidden, utt_context_hidden

    def predict(self, X_da, X_utt, da_encoder, da_decoder, da_context, da_context_hidden,
                utt_encoder, utt_context, utt_context_hidden, config):
        encoder_hidden=da_encoder(X_da)
        da_context_output, da_context_hidden = da_context(encoder_hidden, da_context_hidden)

        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(1, device)
            seq_len = X_utt.size()[1]
            for ei in range(seq_len):
                utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt[ei], utt_encoder_hidden) # (1, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2) # (1, 1, DEC_HIDDEN)
        elif config['use_uttcontext']:
            utt_encoder_hidden = utt_encoder.initHidden(1, device)
            seq_len = X_utt.size()[1]
            for ei in range(seq_len):
                utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt[ei], utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
            utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (1, 1, DEC_HIDDEN)
        else:
            dec_hidden = da_context_output

        decoder_output = da_decoder(dec_hidden)
        decoder_output = decoder_output

        return decoder_output, da_context_hidden, utt_context_hidden
