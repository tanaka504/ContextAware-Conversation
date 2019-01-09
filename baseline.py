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

parser = argparse.ArgumentParser()
parser.add_argument('--expr', '-e', default='baseline', help='input experiment config')
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

def preprocess(X):
    result_x = []
    result_turn = []
    for x_conv in X:
        tmp_x = []
        turn = []
        for x_seq in x_conv:
            if x_seq == '<turn>':
                turn[-1] = 1
            else:
                turn.append(0)
                tmp_x.append(x_seq)
        assert len(tmp_x) == len(turn)
        result_x.append(tmp_x)
        result_turn.append(turn)
    return result_x, result_turn


def make_batchidx(X):
    length = {}
    for idx, conv in enumerate(X):
        if len(conv) in length:
            length[len(conv)].append(idx)
        else:
            length[len(conv)] = [idx]
    return [v for k, v in sorted(length.items(), key=lambda x: x[0])]

def train(experiment):
    print('loading setting "{}"...'.format(experiment))
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, _, _ = create_DAdata(config)
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
    Y_train, _ = preprocess(Y_train)
    Y_valid, _ = preprocess(Y_valid)
    XU_train, YU_train = utt_vocab.tokenize(XU_train, YU_train)
    XU_valid, YU_valid = utt_vocab.tokenize(XU_valid, YU_valid)
    XU_train, Tturn = preprocess(XU_train)
    XU_valid, Vturn = preprocess(XU_valid)

    print('Finish preparing dataset...')

    assert len(X_train) == len(Y_train), 'Unexpect content in train data'
    assert len(X_valid) == len(Y_valid), 'Unexpect content in valid data'
    assert len(Tturn) == len(Y_train), 'turn content invalid shape'

    lr = config['lr']
    batch_size = config['BATCH_SIZE']
    plot_losses = []

    print_total_loss = 0
    plot_total_loss = 0

    da_decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'], da_hidden=config['DEC_HIDDEN']).to(device)
    da_decoder_opt = optim.Adam(da_decoder.parameters(), lr=lr)

    utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
    utt_encoder_opt = optim.Adam(utt_encoder.parameters(), lr=lr)

    utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
    utt_context_opt = optim.Adam(utt_context.parameters(), lr=lr)
    model = baseline().to(device)
    print('Success construct model...')


    criterion = nn.CrossEntropyLoss()

    print('---start training---')

    start = time.time()
    k = 0
    _valid_loss = None
    for e in range(config['EPOCH']):
        tmp_time = time.time()
        print('Epoch {} start'.format(e+1))

        # TODO: 同じターン数でバッチ生成
        indexes = [i for i in range(len(X_train))]
        random.shuffle(indexes)
        k = 0
        while k < len(indexes):
            # initialize
            step_size = min(batch_size, len(indexes) - k)
            utt_context_hidden = utt_context.initHidden(step_size, device)
            da_decoder_opt.zero_grad()
            utt_encoder_opt.zero_grad()
            utt_context_opt.zero_grad()

            batch_idx = indexes[k : k + step_size]

            #  create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(X_train)), end='')
            Y_seq = [Y_train[seq_idx] for seq_idx in batch_idx]
            turn_seq = [Tturn[seq_idx] for seq_idx in batch_idx]
            max_conv_len = max(len(s) + 1 for s in Y_seq)  # seq_len は DA と UTT で共通

            XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]

            # conversation turn padding
            for ci in range(len(XU_seq)):
                XU_seq[ci] = XU_seq[ci] + [[utt_vocab.word2id['<ConvPAD>']]] * (max_conv_len - len(XU_seq[ci]))
            # X_seq  = (batch_size, max_conv_len)
            # XU_seq = (batch_size, max_conv_len, seq_len)

            # conversation turn padding
            for ci in range(len(Y_seq)):
                Y_seq[ci] = Y_seq[ci] + [da_vocab.word2id['<PAD>']] * (max_conv_len - len(Y_seq[ci]))
                turn_seq[ci] = turn_seq[ci] + [0] * (max_conv_len - len(turn_seq[ci]))

            for i in range(0, max_conv_len):
                Y_tensor = torch.tensor([[Y[i]] for Y in Y_seq]).to(device)
                turn_tensor = torch.tensor([[t[i]] for t in turn_seq]).to(device)
                max_seq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<UttPAD>']] * (max_seq_len - len(XU_seq[ci][i]))
                XU_tensor = torch.tensor([XU[i] for XU in XU_seq]).to(device)
                # YU_tensor = torch.tensor([YU[i] for YU in YU_seq]).to(device)
                YU_tensor = None


                # X_tensor = (batch_size, 1)
                # XU_tensor = (batch_size, 1, seq_len)

                last = True if i == len(Y_seq) - 1 else False
    
                if last:
                    loss, utt_context_hidden = model.forward(Y_da=Y_tensor, X_utt=XU_tensor,
                                                             turn=turn_tensor, step_size=step_size,
                                                             da_decoder=da_decoder,
                                                             utt_encoder=utt_encoder, utt_context=utt_context,
                                                             utt_context_hidden=utt_context_hidden,
                                                             criterion=criterion, last=last, config=config)
                    print_total_loss += loss
                    plot_total_loss += loss
                    da_decoder_opt.step()
                    utt_encoder_opt.step()
                    utt_context_opt.step()

                else:
                    utt_context_hidden = model.forward(Y_da=Y_tensor, X_utt=XU_tensor,
                                                       turn=turn_tensor, step_size=step_size,
                                                       da_decoder=da_decoder,
                                                       utt_encoder=utt_encoder, utt_context=utt_context,
                                                       utt_context_hidden=utt_context_hidden,
                                                       criterion=criterion, last=last, config=config)
            k += step_size
        print()
        valid_loss = validation(Y_valid=Y_valid, XU_valid=XU_valid, Vturn=Vturn,
                                model=model, da_decoder=da_decoder,
                                utt_encoder=utt_encoder, utt_context=utt_context, config=config)

        if _valid_loss is None:
            _valid_loss = valid_loss
        else:
            if _valid_loss > valid_loss:
                torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_beststate.model'))
                torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_beststate.model'))
                torch.save(utt_context.state_dict(), os.path.join(config['log_dir'], 'utt_context_beststate.model'))
                _valid_loss = valid_loss


        if (e + 1) % config['LOGGING_FREQ'] == 0:
            print_loss_avg = print_total_loss / config['LOGGING_FREQ']
            print_total_loss = 0
            print('steps %d\tloss %.4f\tvalid loss %.4f | exec time %.4f' % (e + 1, print_loss_avg, valid_loss, time.time() - tmp_time))
            plot_loss_avg = plot_total_loss / config['LOGGING_FREQ']
            plot_losses.append(plot_loss_avg)
            plot_total_loss = 0

        if (e + 1) % config['SAVE_MODEL'] == 0:
            print('saving model')
            torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_state{}.model'.format(e + 1)))
            torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(e + 1)))
            torch.save(utt_context.state_dict(), os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(e + 1)))


    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))

def validation(Y_valid, XU_valid, Vturn, model,
               da_decoder,
               utt_encoder, utt_context, config):

    utt_context_hidden = utt_context.initHidden(1, device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    k = 0

    for seq_idx in range(len(Y_valid)):
        Y_seq = Y_valid[seq_idx]
        XU_seq = XU_valid[seq_idx]
        turn_seq = Vturn[seq_idx]

        for i in range(0, len(Y_seq)):
            Y_tensor = torch.tensor([[Y_seq[i]]]).to(device)
            turn_tensor = torch.tensor([[turn_seq[i]]]).to(device)
            XU_tensor = torch.tensor([XU_seq[i]]).to(device)

            loss, utt_context_hidden = model.evaluate(Y_da=Y_tensor, X_utt=XU_tensor,
                                                  turn=turn_tensor, da_decoder=da_decoder,
                                                  utt_encoder=utt_encoder, utt_context=utt_context,
                                                  utt_context_hidden=utt_context_hidden,
                                                  criterion=criterion, config=config)
            total_loss += loss
    return total_loss

if __name__ == '__main__':
    train(args.expr)
