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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--expr', default='DAonly')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
args = parser.parse_args()

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
    encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_beststate.model')))
    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_beststate.model')))
    context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'context_beststate.model')))

    utt_encoder = None
    utt_context = None
    utt_decoder = None
    if config['use_utt']:
        utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
        utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_beststate.model')))
    if config['use_uttcontext']:
        utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_HIDDEN']).to(device)
        utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_beststate.model')))

    model = DApredictModel().to(device)

    da_context_hidden = context.initHidden(1, device)
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None

    true = []
    pred = []

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
            true.append(Y_tensor.item())
            pred.append(pred_idx.item())

    print()
    true_detok = [da_vocab.id2word[label] for label in true]
    pred_detok = [da_vocab.id2word[label] for label in pred]

    return true, pred, true_detok, pred_detok

def calc_average(y_true, y_pred, average):
    p = precision_score(y_true=y_true, y_pred=y_pred, average=average)
    r = recall_score(y_true=y_true, y_pred=y_pred, average=average)
    f = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    return p, r, f


def save_cmx(y_true, y_pred, expr):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(18, 15))
    plt.rcParams['font.size'] = 18
    sns.heatmap(df_cmx, annot=False)
    plt.xlabel('pre')
    plt.ylabel('next')
    plt.savefig('./data/images/cmx_{}.png'.format(expr))


if __name__ == '__main__':
    # true, pred, true_detok, pred_detok = evaluate(args.expr)
    # c = Counter(true_detok)
    # makefig(X=[k for k in c.keys()], Y=[v/len(true_detok) for v in c.values()],
    #         xlabel='dialogue act', ylabel='freq', imgname='label-freq.png')
    # c = Counter(pred_detok)
    # makefig(X=[k for k in c.keys()], Y=[v/len(true_detok) for v in c.values()],
    #         xlabel='dialogue act', ylabel='pred freq', imgname='predlabel-freq.png')

    # p, r, f = calc_average(true, pred, 'micro')
    # print('p: {}, r: {}, f: {}'.format(p, r, f))
    # acc = accuracy_score(y_true=true_detok, y_pred=pred_detok)
    # print('accuracy: ', acc)
    # save_cmx(true_detok, pred_detok, args.expr)


    config = initialize_env(args.expr)
    preDA, nextDA, _, _, _, _ = create_DAdata(config=config)
    preDA = [label for conv in preDA for label in conv]
    nextDA = [label for conv in nextDA for label in conv]
    save_cmx(y_true=preDA, y_pred=nextDA, expr='bias')

