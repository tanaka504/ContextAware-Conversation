import time
import os
import pyhocon
import torch
from torch import nn
from torch import optim
from models import *
from utils import *
from nn_blocks import *
import argparse
import random
import numpy as np


def train(experiment):
    config = initialize_env(experiment)
    XD_train, YD_train, XU_train, YU_train, turn_train = create_traindata(config=config, prefix='train')
    XD_valid, YD_valid, XU_valid, YU_valid, turn_valid = create_traindata(config=config, prefix='valid')
    print('Finish create train data...')

    if os.path.exists(os.path.join(config['log_root'], 'da_vocab.dict')):
        da_vocab = da_Vocab(config, create_vocab=False)
        utt_vocab = utt_Vocab(config, create_vocab=False)
    else:
        da_vocab = da_Vocab(config, das=[token for conv in XD_train + XD_valid + YD_train + YD_valid for token in conv])
        utt_vocab = utt_Vocab(config,
                              sentences=[sentence for conv in XU_train + XU_valid + YU_train + YU_valid for sentence in
                                         conv])
        da_vocab.save()
        utt_vocab.save()
    print('Utterance vocab.: {}'.format(len(utt_vocab.word2id)))
    print('Dialog Act vocab.: {}'.format(len(da_vocab.word2id)))

    # Tokenize
    XD_train, YD_train = da_vocab.tokenize(XD_train), da_vocab.tokenize(YD_train)
    XD_valid, YD_valid = da_vocab.tokenize(XD_valid), da_vocab.tokenize(YD_valid)
    XU_train, YU_train = utt_vocab.tokenize(XU_train), utt_vocab.tokenize(YU_train)
    XU_valid, YU_valid = utt_vocab.tokenize(XU_valid), utt_vocab.tokenize(YU_valid)
    assert len(XD_train) == len(YD_train), 'Unexpect content in train data'
    assert len(XD_valid) == len(YD_valid), 'Unexpect content in valid data'
    lr = config['lr']
    batch_size = config['BATCH_SIZE']

    predictor = DApredictModel(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config)
    model_opt = optim.Adam(predictor.parameters(), lr=lr)
    start = time.time()
    _valid_loss = None
    _train_loss = None
    total_loss = 0
    early_stop = 0
    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))
        indexes = [i for i in range(len(XD_train))]
        random.shuffle(indexes)
        k = 0
        predictor.train()
        while k < len(indexes):
            # initialize
            step_size = min(batch_size, len(indexes) - k)
            batch_idx = indexes[k: k + step_size]
            model_opt.zero_grad()
            # create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(XD_train)), end='')
            XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]
            XD_seq = [XD_train[seq_idx] for seq_idx in batch_idx]
            YD_seq = [YD_train[seq_idx] for seq_idx in batch_idx]
            turn_seq = [turn_train[seq_idx] for seq_idx in batch_idx]
            max_conv_len = max(len(s) for s in XU_seq)
            XU_tensor = []
            XD_tensor = []
            turn_tensor = []
            for i in range(0, max_conv_len):
                max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
                XU_tensor.append(torch.tensor([XU[i] for XU in XU_seq]).cpu())
                XD_tensor.append(torch.tensor([[XD[i]] for XD in XD_seq]).cpu())
                turn_tensor.append(torch.tensor([[t[i]] for t in turn_seq]).cpu())
            if config['DApred']['predict']:
                XD_tensor = XD_tensor[:-1]
                YD_tensor = torch.tensor([YD[-2] for YD in YD_seq]).cpu()
            else:
                YD_tensor = torch.tensor([YD[-1] for YD in YD_seq]).cpu()
            loss, preds = predictor.forward(X_da=XD_tensor, Y_da=YD_tensor, X_utt=XU_tensor, turn=turn_tensor, step_size=step_size)
            model_opt.step()
            total_loss += loss
            k += step_size
        print()

        valid_loss, valid_acc = validation(XD_valid=XD_valid, XU_valid=XU_valid, YD_valid=YD_valid, turn_valid=turn_valid,
                                           model=predictor, utt_vocab=utt_vocab, config=config)

        def save_model(filename):
            torch.save(predictor.state_dict(), os.path.join(config['log_dir'], 'da_pred_state{}.model'.format(filename)))

        if _valid_loss is None:
            save_model('validbest')
            _valid_loss = valid_loss
        else:
            if _valid_loss > valid_loss:
                save_model('validbest')
                _valid_loss = valid_loss
                print('valid loss update, save model')

        if _train_loss is None:
            save_model('trainbest')
            _train_loss = total_loss
        else:
            if _train_loss > total_loss:
                save_model('trainbest')
                _train_loss = total_loss
                early_stop = 0
                print('train loss update, save model')
            else:
                early_stop += 1
                print('early stopping count | {}/{}'.format(early_stop, config['EARLY_STOP']))
                if early_stop >= config['EARLY_STOP']:
                    break
        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = total_loss / config['LOGGING_FREQ']
            total_loss = 0
            print('steps %d\tloss %.4f\tvalid loss %.4f\tvalid acc %.4f | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, valid_acc, time.time() - tmp_time))

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('saving model')
            save_model(e+1)
    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))


def validation(XD_valid, XU_valid, YD_valid, turn_valid, model, utt_vocab, config):
    model.eval()
    total_loss = 0
    k = 0
    batch_size = config['BATCH_SIZE']
    indexes = [i for i in range(len(XU_valid))]
    acc = []
    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k: k + step_size]
        XU_seq = [XU_valid[seq_idx] for seq_idx in batch_idx]
        XD_seq = [XD_valid[seq_idx] for seq_idx in batch_idx]
        YD_seq = [YD_valid[seq_idx] for seq_idx in batch_idx]
        turn_seq = [turn_valid[seq_idx] for seq_idx in batch_idx]
        max_conv_len = max(len(s) for s in XU_seq)
        XU_tensor = []
        XD_tensor = []
        turn_tensor = []
        for i in range(0, max_conv_len):
            max_xseq_len = max(len(XU[i]) + 1 for XU in XU_seq)
            for ci in range(len(XU_seq)):
                XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<PAD>']] * (max_xseq_len - len(XU_seq[ci][i]))
            XU_tensor.append(torch.tensor([x[i] for x in XU_seq]).cpu())
            XD_tensor.append(torch.tensor([[x[i]] for x in XD_seq]).cpu())
            turn_tensor.append(torch.tensor([[t[i]] for t in turn_seq]).cpu())
        if config['DApred']['predict']:
            XD_tensor = XD_tensor[:-1]
            YD_tensor = torch.tensor([YD[-2] for YD in YD_seq]).cpu()
        else:
            YD_tensor = torch.tensor([YD[-1] for YD in YD_seq]).cpu()
        loss, preds = model(X_da=XD_tensor, Y_da=YD_tensor, X_utt=XU_tensor, turn=turn_tensor, step_size=step_size)
        preds = np.argmax(preds, axis=1)
        acc.append(accuracy_score(y_pred=preds, y_true=YD_tensor.data.tolist()))
        total_loss += loss
        k += step_size
    return total_loss, np.mean(acc)

if __name__ == '__main__':
    args = parse()
    train(args.expr)
