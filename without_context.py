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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('--expr', '-e', default='DAwocontext', help='input experiment config')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = 'cuda'
else:
    device = 'cpu'


def initialize_env(name):
    config = pyhocon.ConfigFactory.parse_file('experiments.conf')[name]
    config['log_dir'] = os.path.join(config['log_root'], name)
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    return config


def create_DAdata(config):
    posts, cmnts, _, _ = create_traindata(config)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = separate_data(posts, cmnts)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def create_Uttdata(config):
    _, _, posts, cmnts = create_traindata(config)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = separate_data(posts, cmnts)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def make_batchidx(X):
    length = {}
    for idx, conv in enumerate(X):
        if len(conv) in length:
            length[len(conv)].append(idx)
        else:
            length[len(conv)] = [idx]
    return [v for k, v in sorted(length.items(), key=lambda x: x[0])]

def flatten(X):
    return [utt for conv in X for utt in conv]


def train(experiment):
    print('loading setting "{}"...'.format(experiment))
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, _, _, Tturn, Vturn,  = create_DAdata(config)
    print('Finish create train data...')
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    if config['use_utt']:
        XU_train, YU_train, XU_valid, YU_valid, _, _ = create_Uttdata(config)
        utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)
    else:
        utt_vocab = None
    print('Finish create vocab dic...')

    # Tokenize sequences
    X_train, Y_train = da_vocab.tokenize(X_train, Y_train)
    X_valid, Y_valid = da_vocab.tokenize(X_valid, Y_valid)
    X_train = flatten(X_train)
    Y_train = flatten(Y_train)
    X_valid = flatten(X_valid)
    Y_valid = flatten(Y_valid)
    if config['use_utt']:
        XU_train, YU_train = utt_vocab.tokenize(XU_train, YU_train)
        XU_valid, YU_valid = utt_vocab.tokenize(XU_valid, YU_valid)
        XU_train = flatten(XU_train)
        YU_train = flatten(YU_train)
        XU_valid = flatten(XU_valid)
        YU_valid = flatten(YU_valid)
    else:
        XU_train = []
        YU_train = []
        XU_valid = []
        YU_valid = []


    print('Finish preparing dataset...')

    assert len(X_train) == len(Y_train), 'Unexpect content in train data'
    assert len(X_valid) == len(Y_valid), 'Unexpect content in valid data'

    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    plot_losses = []

    print_total_loss = 0
    plot_total_loss = 0

    da_encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                           da_hidden=config['DA_HIDDEN']).to(device)
    da_decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                           da_hidden=config['DEC_HIDDEN']).to(device)
    da_encoder_opt = optim.Adam(da_encoder.parameters(), lr=lr)
    da_decoder_opt = optim.Adam(da_decoder.parameters(), lr=lr)

    utt_encoder = None
    utt_decoder = None
    if config['use_utt']:
        utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'],
                                       utterance_hidden=config['UTT_HIDDEN'],
                                       padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
        utt_encoder_opt = optim.Adam(utt_encoder.parameters(), lr=lr)

    model = DAwocontext().to(device)
    print('Success construct model...')

    criterion = nn.CrossEntropyLoss()

    print('---start training---')

    start = time.time()
    k = 0
    _valid_loss = None
    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e + 1))
        indexes = [i for i in range(len(X_train))]
        random.shuffle(indexes)
        k = 0
        while k < len(indexes):
            # initialize
            step_size = min(batch_size, len(indexes) - k)
            da_encoder_opt.zero_grad()
            da_decoder_opt.zero_grad()
            if config['use_utt']:
                utt_encoder_opt.zero_grad()

            batch_idx = indexes[k: k + step_size]

            # create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(X_train)), end='')
            X_seq = [X_train[seq_idx] for seq_idx in batch_idx]
            Y_seq = [Y_train[seq_idx] for seq_idx in batch_idx]

            if config['use_utt']:
                XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]
                YU_seq = [YU_train[seq_idx] for seq_idx in batch_idx]

            assert len(X_seq) == len(Y_seq), 'Unexpect sequence length'

            X_tensor = torch.tensor(X_seq).to(device)
            Y_tensor = torch.tensor(Y_seq).to(device)
            if config['use_utt']:
                max_seq_len = max(len(XU) + 1 for XU in XU_seq)
                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci] = XU_seq[ci] + [utt_vocab.word2id['<UttPAD>']] * (
                            max_seq_len - len(XU_seq[ci]))
                    YU_seq[ci] = YU_seq[ci] + [utt_vocab.word2id['<UttPAD>']] * (
                            max_seq_len - len(YU_seq[ci]))
                XU_tensor = torch.tensor(XU_seq).to(device)
                # YU_tensor = torch.tensor([YU[i] for YU in YU_seq]).to(device)
                YU_tensor = None

            else:
                XU_tensor, YU_tensor = None, None

            # X_tensor = (batch_size, 1)
            # XU_tensor = (batch_size, 1, seq_len)

            loss = model.forward(X_da=X_tensor, Y_da=Y_tensor,
                                 X_utt=XU_tensor,
                                 step_size=step_size,
                                 da_encoder=da_encoder,
                                 da_decoder=da_decoder,
                                 utt_encoder=utt_encoder,
                                 criterion=criterion,
                                 config=config)
            print_total_loss += loss
            plot_total_loss += loss
            da_encoder_opt.step()
            da_decoder_opt.step()
            if config['use_utt']:
                utt_encoder_opt.step()

            k += step_size
        print()
        valid_loss = validation(X_valid=X_valid, Y_valid=Y_valid, XU_valid=XU_valid, YU_valid=YU_valid,
                                model=model, da_encoder=da_encoder, da_decoder=da_decoder,
                                utt_encoder=utt_encoder, utt_decoder=utt_decoder,
                                config=config)

        if _valid_loss is None:
            _valid_loss = valid_loss
        else:
            if _valid_loss > valid_loss:
                torch.save(da_encoder.state_dict(), os.path.join(config['log_dir'], 'enc_beststate.model'))
                torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_beststate.model'))
                if config['use_utt']:
                    torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_beststate.model'))
                _valid_loss = valid_loss

        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = print_total_loss / config['LOGGING_FREQ']
            print_total_loss = 0
            print('steps %d\tloss %.4f\tvalid loss %.4f | exec time %.4f' % (
            e + 1, print_loss_avg, valid_loss, time.time() - tmp_time))
            plot_loss_avg = plot_total_loss / config['LOGGING_FREQ']
            plot_losses.append(plot_loss_avg)
            plot_total_loss = 0

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('saving model')
            torch.save(da_encoder.state_dict(), os.path.join(config['log_dir'], 'enc_state{}.model'.format(e + 1)))
            torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_state{}.model'.format(e + 1)))
            if config['use_utt']:
                torch.save(utt_encoder.state_dict(),
                           os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(e + 1)))

    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))


def validation(X_valid, Y_valid, XU_valid, YU_valid, model,
               da_encoder, da_decoder,
               utt_encoder, utt_decoder, config):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    k = 0

    for seq_idx in range(len(X_valid)):
        X_seq = X_valid[seq_idx]
        Y_seq = Y_valid[seq_idx]
        if config['use_utt']:
            XU_seq = XU_valid[seq_idx]
            YU_seq = YU_valid[seq_idx]

        X_tensor = torch.tensor([X_seq]).to(device)
        Y_tensor = torch.tensor([Y_seq]).to(device)
        if config['use_utt']:
            XU_tensor = torch.tensor([XU_seq]).to(device)
            # YU_tensor = torch.tensor([YU_seq]).to(device)
        else:
            XU_tensor, YU_tensor = None, None

        loss = model.evaluate(X_da=X_tensor, Y_da=Y_tensor, X_utt=XU_tensor,
                              da_encoder=da_encoder, da_decoder=da_decoder,
                              utt_encoder=utt_encoder,
                              criterion=criterion, config=config)
    total_loss += loss
    return total_loss

def evaluation(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = create_DAdata(config)
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    if config['use_utt']:
        XU_train, YU_train, XU_valid, YU_valid, XU_test, YU_test = create_Uttdata(config)
        utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)

    X_test, Y_test = da_vocab.tokenize(X_test, Y_test)
    X_test = flatten(X_test)
    Y_test = flatten(Y_test)
    if config['use_utt']:
        XU_test, _ = utt_vocab.tokenize(XU_test, YU_test)
        XU_test = flatten(XU_test)
    else:
        XU_test = []

    print('load models')
    encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DA_HIDDEN']).to(device)
    decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DEC_HIDDEN']).to(device)


    # loading weight
    encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'enc_beststate.model')))
    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_beststate.model')))

    utt_encoder = None
    utt_context = None
    if config['use_utt']:
        utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'],
                                       utterance_hidden=config['UTT_HIDDEN'],
                                       padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
        utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_beststate.model')))
    if config['use_uttcontext']:
        utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_HIDDEN']).to(device)
        utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_beststate.model')))

    model = DAwocontext().to(device)

    true = []
    pred = []

    X_seq = X_test
    Y_seq = Y_test
    if config['use_utt']:
        XU_seq = XU_test
    assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in test data'

    for i in range(0, len(X_seq)):
        X_tensor = torch.tensor([X_seq[i]]).to(device)
        Y_tensor = torch.tensor(Y_seq[i]).to(device)
        if config['use_utt']:
            XU_tensor = torch.tensor([XU_seq[i]]).to(device)
        else:
            XU_tensor = None

        decoder_output = model.predict(X_da=X_tensor, X_utt=XU_tensor,
                                      da_encoder=encoder,
                                      da_decoder=decoder,
                                      utt_encoder=utt_encoder,
                                      config=config)
        pred_idx = torch.argmax(decoder_output)
        true.append(Y_tensor.item())
        pred.append(pred_idx.item())

    print()
    true_detok = [da_vocab.id2word[label] for label in true]
    pred_detok = [da_vocab.id2word[label] for label in pred]

    return true, pred, true_detok, pred_detok

def calc_average(y_true, y_pred):
    p = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    r = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    f = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('p: {} | r: {} | f: {} | acc: {}'.format(p, r, f, acc))


if __name__ == '__main__':
    # train(args.expr)
    true, pred, true_detok, pred_detok = evaluation(args.expr)
    calc_average(y_true=true, y_pred=pred)