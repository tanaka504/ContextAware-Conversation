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
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pprint import pprint
import numpy as np
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--expr', default='DAonly')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
args = parser.parse_args()

def evaluate(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, _, _, turn = create_DAdata(config)
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    XU_train, YU_train, XU_valid, YU_valid, XU_test, YU_test = create_Uttdata(config)
    utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)

    X_test, Y_test = da_vocab.tokenize(X_test, Y_test)
    # X_test, turn = preprocess(X_test)
    # Y_test, _ = preprocess(Y_test)
    XU_test, _ = utt_vocab.tokenize(XU_test, YU_test)
    # XU_test, _ = preprocess(XU_test)

    print('load models')
    encoder, context = None, None
    if config['use_da']:
        encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
        context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
        encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_beststate.model')))
        context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'context_beststate.model')))

    decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DEC_HIDDEN']).to(device)

    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_beststate.model')))

    utt_encoder = None
    utt_context = None
    utt_decoder = None
    if config['use_utt'] or config['use_uttcontext']:
        utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
        utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_beststate.model')))
    if config['use_uttcontext']:
        utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
        utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_beststate.model')))

    model = DApredictModel(device).to(device)

    da_context_hidden = context.initHidden(1, device) if config['use_da'] else None
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None

    result = []

    for seq_idx in range(0, len(X_test)):
        print('\r{}/{} conversation evaluating'.format(seq_idx+1, len(X_test)), end='')
        X_seq = X_test[seq_idx]
        Y_seq = Y_test[seq_idx]
        turn_seq = turn[seq_idx]
        XU_seq = XU_test[seq_idx]
        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in test data'

        pred_seq = []
        true_seq = []

        for i in range(0, len(X_seq)):
            X_tensor = torch.tensor([[X_seq[i]]]).to(device)
            Y_tensor = torch.tensor(Y_seq[i]).to(device)
            if config['turn']:
                turn_tensor = torch.tensor([[turn_seq[i]]]).to(device)
            else:
                turn_tensor = None
            if config['use_utt'] or config['use_uttcontext']:
                XU_tensor = torch.tensor([[XU_seq[i]]]).to(device)
            else:
                XU_tensor = None

            decoder_output, da_context_hidden, utt_context_hidden = model.predict(X_da=X_tensor, X_utt=XU_tensor,
                                                           turn=turn_tensor,
                                                           da_encoder=encoder, da_decoder=decoder, da_context=context,
                                                           da_context_hidden=da_context_hidden,
                                                           utt_encoder=utt_encoder, utt_context=utt_context,
                                                           utt_context_hidden=utt_context_hidden, config=config)
            pred_idx = torch.argmax(decoder_output)
            pred_seq.append(pred_idx.item())
            true_seq.append(Y_tensor.item())
        result.append({'true': true_seq,
                       'true_detok': [da_vocab.id2word[token] for token in true_seq],
                       'pred': pred_seq,
                       'pred_detok': [da_vocab.id2word[token] for token in pred_seq],
                       'UttSeq': [[utt_vocab.id2word[word] for word in sentence] for sentence in XU_seq],
                       'seq_detok': [da_vocab.id2word[label] for label in X_seq]})

    print()

    return result, da_vocab

def calc_average(y_true, y_pred):
    p = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    r = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    f = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('p: {} | r: {} | f: {} | acc: {}'.format(p, r, f, acc))


def save_cmx(y_true, y_pred, expr):
    fontsize = 40
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(40, 30))
    plt.rcParams['font.size'] = fontsize
    heatmap = sns.heatmap(df_cmx, annot=True, fmt='d', cmap='Blues')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('./data/images/cmx_{}.png'.format(expr))


if __name__ == '__main__':
    result, da_vocab = evaluate(args.expr)
    with open('./data/images/{}.result'.format(args.expr), 'wb') as f:
        pickle.dump(result, f)
    # noturn_idx = [idx for idx, token in enumerate(true_detok) if not token == '<turn>']
    # c = Counter(true_detok)
    # makefig(X=[k for k in c.keys()], Y=[v/len(true_detok) for v in c.values()],
    #         xlabel='dialogue act', ylabel='freq', imgname='label-freq.png')
    # c = Counter(pred_detok)
    # makefig(X=[k for k in c.keys()], Y=[v/len(true_detok) for v in c.values()],
    #         xlabel='dialogue act', ylabel='pred freq', imgname='predlabel-freq.png')
    # c = Counter(noturn_pred)

    # noturn_true = [true[idx] for idx in noturn_idx]
    # noturn_pred = [pred[idx] for idx in noturn_idx]

    true = [label for line in result for label in line['true']]
    pred = [label for line in result for label in line['pred']]


    calc_average(y_true=true, y_pred=pred)
    f = f1_score(y_true=true, y_pred=pred, average=None)
    [print(da_vocab.id2word[idx], score) for idx, score in zip(sorted(set(true)),f)]

    # save_cmx(y_true=[line['true_detok'] for line in result], y_pred=[line['pred_detok'] for line in result], expr=args.expr)
    # config = initialize_env('DAonlyturn')
    # preDA, nextDA, _, _, _ = create_traindata(config)
    # c = Counter([label for conv in preDA for label in conv])
    # pprint({k: v for k, v in c.items()})
    # print(len(preDA), sum([len(conv) for conv in preDA]))
    # tmp = [('->'.join(conv[:-1]), conv[-1]) for conv in preDA]
    # save_cmx(y_true=[pre for seq in preDA for pre in seq], y_pred=[nex for seq in nextDA for nex in seq], expr='DApattern')


    # print(sum([len(conv) for conv in preDA])/ len(preDA))
    # preDA, turn = preprocess(preDA, mode='X')
    # nextDA, _ = preprocess(nextDA, mode='Y')
    # assert len(preDA) == len(nextDA)
    # print('Conversations: ', len(preDA))
    # preDA = [label for conv in preDA for label in conv]
    # nextDA = {label for conv in nextDA for label in conv}
    # print(nextDA)
    # print('Sentences: ', len(preDA))
    # c = Counter(nextDA)
    # pprint({k: v for k, v in c.items()})
    # save_cmx(y_true=preDA, y_pred=nextDA, expr='bias')

