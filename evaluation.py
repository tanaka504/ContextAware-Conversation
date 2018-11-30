import time
import os
import pyhocon
import torch
from torch import optim
from models import *
from nn_blocks import *
from utils import *
from train import initialize_env, create_DAdata, create_Uttdata, device
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expr', default='DAonly')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
args = parser.parse_args()

model_num = 8

def evaluate(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = create_DAdata(config)
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    if config['use_utt']:
        XU_train, YU_train, XU_valid, YU_valid, XU_test, YU_test = create_Uttdata(config)
        utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)

    X_test, Y_test = da_vocab.tokenize(X_test, Y_test)
    if config['use_utt']:
        XU_test, _ = utt_vocab.tokenize(XU_test, YU_test)
    else:
        XU_test = []

    print('load models')
    encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
    decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DEC_HIDDEN']).to(device)
    context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)

    # loading weight
    encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_state{}.model'.format(model_num))))
    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_state{}.model'.format(model_num))))
    context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'context_state{}.model'.format(model_num))))

    utt_encoder = None
    utt_context = None
    utt_decoder = None
    if config['use_utt']:
        utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
        utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(model_num))))
    if config['use_uttcontext']:
        utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_HIDDEN']).to(device)
        utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(model_num))))

    model = DApredictModel().to(device)

    da_context_hidden = context.initHidden(1, device)
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None

    correct = 0

    for seq_idx in range(0, len(X_test)):
        print('\r{}/{} sequences evaluating'.format(seq_idx+1, len(X_test)), end='')
        X_seq = X_test[seq_idx]
        Y_seq = Y_test[seq_idx]
        if config['use_utt']:
            XU_seq = XU_test[seq_idx]
        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in test data'

        for i in range(0, len(X_seq)):
            X_tensor = torch.tensor([[X_seq[i]]]).to(device)
            Y_tensor = torch.tensor(Y_seq[i]).to(device)
            if config['use_utt']:
                XU_tensor = torch.tensor([[XU_seq[i]]]).to(device)
            else:
                XU_tensor = None

            decoder_output, da_context_hidden, utt_context_hidden = model.predict(X_da=X_tensor, X_utt=XU_tensor,
                                                           da_encoder=encoder, da_decoder=decoder, da_context=context,
                                                           da_context_hidden=da_context_hidden,
                                                           utt_encoder=utt_encoder, utt_context=utt_context,
                                                           utt_context_hidden=utt_context_hidden, config=config)
            pred_idx = torch.argmax(decoder_output)
            if pred_idx.item() == Y_tensor.item(): correct += 1
    print()

    return correct / len(X_test)

if __name__ == '__main__':
    true_rate = evaluate(args.expr)
    print(true_rate)