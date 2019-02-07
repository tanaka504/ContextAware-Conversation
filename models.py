import torch
import torch.nn as nn
from nn_blocks import *
from torch import optim
import time


class DApredictModel(nn.Module):
    def __init__(self, device):
        super(DApredictModel, self).__init__()
        self.device = device

    def forward(self, X_da, Y_da, X_utt, Y_utt, step_size, turn,
                da_encoder, da_decoder, da_context, da_context_hidden,
                utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                criterion, last, config):
        loss = 0
        if config['use_da']:
            da_encoder_hidden = da_encoder(X_da) # (batch_size, 1, DA_HIDDEN)
            da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden) # (batch_size, 1, DA_HIDDEN)
        if config['turn']:
            turn = turn.float()
            turn = turn.unsqueeze(1)  # (batch_size, 1, 1)


        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(step_size, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden) # (batch_size, 1, UTT_HIDDEN)
            if config['turn']:
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_encoder_output, turn), dim=2) # (batch_size, 1, DEC_HIDDEN)
                else:
                    dec_hidden = torch.cat((utt_encoder_output, turn), dim=2)
            else:
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2)
                else:
                    dec_hidden = utt_encoder_output
        elif config['use_uttcontext']:
            utt_encoder_hidden = utt_encoder.initHidden(step_size, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)
            if config['turn']:
                utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)
            utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)
            if config['use_da']:
                dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (batch_size, 1, DEC_HIDDEN)
                if not config['use_dacontext']:
                    dec_hiden = torch.cat((da_encoder_hidden, utt_context_output), dim=2)
            else:
                dec_hidden = utt_context_output
        else:
            if config['turn']:
                dec_hidden = torch.cat((da_context_output, turn), dim=2)
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

    def evaluate(self, X_da, Y_da, X_utt, Y_utt, turn,
                 da_encoder, da_decoder, da_context, da_context_hidden,
                 utt_encoder, utt_decoder, utt_context, utt_context_hidden,
                 criterion, config):
        with torch.no_grad():
            loss = 0
            if config['use_da']:
                da_encoder_hidden = da_encoder(X_da)
                da_context_output, da_context_hidden = da_context(da_encoder_hidden, da_context_hidden)

            if config['turn']:
                turn = turn.float()
                turn = turn.unsqueeze(1)  # (batch_size, 1, 1)

            if config['use_utt']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden) # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output, turn), dim=2) # (1, 1, DEC_HIDDEN)
                    else:
                        dec_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                else:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2)
                    else:
                        dec_hidden = utt_encoder_output
            elif config['use_uttcontext']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (1, 1, DEC_HIDDEN)
                    if not config['use_dacontext']:
                        dec_hidden = torch.cat((da_encoder_hidden, utt_context_output), dim=2)
                else:
                    dec_hidden = utt_context_output
            else:
                if config['turn']:
                    dec_hidden = torch.cat((da_context_output, turn), dim=2)
                else:
                    dec_hidden = da_context_output

            decoder_output = da_decoder(dec_hidden)
            decoder_output = decoder_output.squeeze(1)
            Y_da = Y_da.squeeze(0)

            loss += criterion(decoder_output, Y_da)

        return loss.item(), da_context_hidden, utt_context_hidden

    def predict(self, X_da, X_utt, turn, da_encoder, da_decoder, da_context, da_context_hidden,
                utt_encoder, utt_context, utt_context_hidden, config):
        with torch.no_grad():
            if config['use_da']:
                encoder_hidden=da_encoder(X_da)
                da_context_output, da_context_hidden = da_context(encoder_hidden, da_context_hidden)
        
            if config['turn']:
                turn = turn.float()
                turn = turn.unsqueeze(1)  # (batch_size, 1, 1)

            if config['use_utt']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                seq_len = X_utt.size()[1]
                for ei in range(seq_len):
                    utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt[ei], utt_encoder_hidden) # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output, turn), dim=2) # (1, 1, DEC_HIDDEN)
                    else:
                        dec_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                else:
                    if config['use_da']:
                        dec_hidden = torch.cat((da_context_output, utt_encoder_output), dim=2)
                    else:
                        dec_hidden = utt_encoder_output

            elif config['use_uttcontext']:
                utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
                seq_len = X_utt.size()[1]
                for ei in range(seq_len):
                    utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt[ei], utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
                if config['turn']:
                    utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)
                utt_context_output, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)
                if config['use_da']:
                    dec_hidden = torch.cat((da_context_output, utt_context_output), dim=2) # (1, 1, DEC_HIDDEN)
                    if not config['use_dacontext']:
                        dec_hidden = torch.cat((encoder_hidden, utt_context_output), dim=2)
                else:
                    dec_hidden = utt_context_output
            else:
                if config['turn']:
                    dec_hidden = torch.cat((da_context_output, turn), dim=2)
                else:
                    dec_hidden = da_context_output

            decoder_output = da_decoder(dec_hidden)

        return decoder_output, da_context_hidden, utt_context_hidden

class DAwocontext(nn.Module):
    def __init__(self, device):
        super(DAwocontext, self).__init__()
        self.device = device

    def forward(self, X_da, Y_da, X_utt, step_size,
                da_encoder, utt_encoder, da_decoder, criterion, config):

        da_encoder_hidden = da_encoder(X_da) # (batch_size, 1, DA_HIDDEN)
        da_encoder_hidden = da_encoder_hidden.unsqueeze(1)

        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(step_size, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden) # (batch_size, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_encoder_hidden, utt_encoder_output), dim=2) # (batch_size, 1, DEC_HIDDEN)
        else:
            dec_hidden = da_encoder_hidden


        decoder_output = da_decoder(dec_hidden) # (batch_size, 1, DA_VOCAB)
        decoder_output = decoder_output.squeeze(1) # (batch_size, DA_VOCAB)
        Y_da = Y_da.squeeze()


        loss = criterion(decoder_output, Y_da)

        loss.backward()

        return loss

    def evaluate(self, X_da, Y_da, X_utt,
                da_encoder, utt_encoder, da_decoder, criterion, config):

        da_encoder_hidden = da_encoder(X_da)  # (batch_size, 1, DA_HIDDEN)
        da_encoder_hidden = da_encoder_hidden.unsqueeze(1)

        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt,
                                                                 utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_encoder_hidden, utt_encoder_output), dim=2)  # (batch_size, 1, DEC_HIDDEN)
        else:
            dec_hidden = da_encoder_hidden

        decoder_output = da_decoder(dec_hidden)  # (batch_size, 1, DA_VOCAB)
        decoder_output = decoder_output.squeeze(1)  # (batch_size, DA_VOCAB)
        # Y_da = Y_da.squeeze()

        loss = criterion(decoder_output, Y_da)

        return loss

    def predict(self, X_da, X_utt,
                da_encoder, utt_encoder, da_decoder, config):

        da_encoder_hidden = da_encoder(X_da)  # (batch_size, 1, DA_HIDDEN)
        da_encoder_hidden = da_encoder_hidden.unsqueeze(1)

        if config['use_utt']:
            utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt,
                                                                 utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)
            dec_hidden = torch.cat((da_encoder_hidden, utt_encoder_output), dim=2)  # (batch_size, 1, DEC_HIDDEN)
        else:
            dec_hidden = da_encoder_hidden

        decoder_output = da_decoder(dec_hidden)  # (batch_size, 1, DA_VOCAB)

        return decoder_output

class baseline(nn.Module):
    def __init__(self, device):
        super(baseline, self).__init__()
        self.device = device

    def forward(self, Y_da, X_utt, turn, step_size,
                da_decoder, utt_encoder, utt_context, utt_context_hidden,
                criterion, last, config):
        loss = 0

        utt_encoder_hidden = utt_encoder.initHidden(step_size, self.device)
        utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (batch_size, 1, UTT_HIDDEN)

        turn = turn.unsqueeze(1) # (batch_size, 1, 1)

        utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2) # (batch_size, 1, UTT_HIDDEN)
        dec_hidden, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (batch_size, 1, UTT_HIDDEN)

        decoder_output = da_decoder(dec_hidden) # (batch_size, 1, DA_VOCAB)
        decoder_output = decoder_output.squeeze(1) # (batch_size, DA_VOCAB)
        Y_da = Y_da.squeeze()

        loss += criterion(decoder_output, Y_da)

        if last:
            loss.backward()
            return loss.item(), utt_context_hidden
        else:
            return utt_context_hidden

    def evaluate(self, Y_da, X_utt, turn,
                 da_decoder,
                 utt_encoder, utt_context, utt_context_hidden,
                 criterion, config):
        loss = 0

        utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
        utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt, utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
        turn = turn.unsqueeze(1)  # (batch_size, 1, 1)
        utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)  # (batch_size, 1, UTT_HIDDEN)
        dec_hidden, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)

        decoder_output = da_decoder(dec_hidden)
        decoder_output = decoder_output.squeeze(1)
        Y_da = Y_da.squeeze(0)

        loss += criterion(decoder_output, Y_da)

        return loss.item(), utt_context_hidden

    def predict(self, X_utt, turn, da_decoder,
                utt_encoder, utt_context, utt_context_hidden, config):
        utt_encoder_hidden = utt_encoder.initHidden(1, self.device)
        seq_len = X_utt.size()[1]
        for ei in range(seq_len):
            utt_encoder_output, utt_encoder_hidden = utt_encoder(X_utt[ei], utt_encoder_hidden)  # (1, 1, UTT_HIDDEN)
        turn = turn.float()
        turn = turn.unsqueeze(1)  # (batch_size, 1, 1)
        utt_encoder_hidden = torch.cat((utt_encoder_output, turn), dim=2)  # (batch_size, 1, UTT_HIDDEN)
        dec_hidden, utt_context_hidden = utt_context(utt_encoder_hidden, utt_context_hidden) # (1, 1, UTT_HIDDEN)

        decoder_output = da_decoder(dec_hidden)

        return decoder_output, utt_context_hidden




