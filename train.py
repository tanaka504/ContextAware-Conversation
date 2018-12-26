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
parser.add_argument('--expr', '-e', default='DAonly', help='input experiment config')
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
    if config['use_utt']:
        XU_train, YU_train = utt_vocab.tokenize(XU_train, YU_train)
        XU_valid, YU_valid = utt_vocab.tokenize(XU_valid, YU_valid)
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

    da_encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'], da_hidden=config['DA_HIDDEN']).to(device)
    da_decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'], da_hidden=config['DEC_HIDDEN']).to(device)
    da_context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
    da_encoder_opt = optim.Adam(da_encoder.parameters(), lr=lr)
    da_decoder_opt = optim.Adam(da_decoder.parameters(), lr=lr)
    da_context_opt = optim.Adam(da_context.parameters(), lr=lr)

    utt_encoder = None
    utt_context = None
    utt_decoder = None
    if config['use_utt']:
        utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
        utt_encoder_opt = optim.Adam(utt_encoder.parameters(), lr=lr)
    if config['use_uttcontext']:
        utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_HIDDEN']).to(device)
        utt_context_opt = optim.Adam(utt_context.parameters(), lr=lr)
    model = DApredictModel().to(device)
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
            context_hidden = da_context.initHidden(step_size, device)
            utt_context_hidden = utt_context.initHidden(step_size, device) if config['use_uttcontext'] else None
            da_context_opt.zero_grad()
            da_encoder_opt.zero_grad()
            da_decoder_opt.zero_grad()
            if config['use_utt']:
                utt_encoder_opt.zero_grad()
            if config['use_uttcontext']:
                utt_context_opt.zero_grad()

            batch_idx = indexes[k : k + step_size]

            #  create batch data
            print('\rConversation {}/{} training...'.format(k + step_size, len(X_train)), end='')
            X_seq = [X_train[seq_idx] for seq_idx in batch_idx]
            Y_seq = [Y_train[seq_idx] for seq_idx in batch_idx]
            max_conv_len = max(len(s) + 1 for s in X_seq)  # seq_len は DA と UTT で共通
            if config['use_utt']:
                XU_seq = [XU_train[seq_idx] for seq_idx in batch_idx]
                YU_seq = [YU_train[seq_idx] for seq_idx in batch_idx]

                # conversation turn padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci] = XU_seq[ci] + [[utt_vocab.word2id['<ConvPAD>']]] * (max_conv_len - len(XU_seq[ci]))
                    YU_seq[ci] = YU_seq[ci] + [[utt_vocab.word2id['<ConvPAD>']]] * (max_conv_len - len(YU_seq[ci]))
            # X_seq  = (batch_size, max_conv_len)
            # XU_seq = (batch_size, max_conv_len, seq_len)

            # conversation turn padding
            for ci in range(len(X_seq)):
                X_seq[ci] = X_seq[ci] + [da_vocab.word2id['<PAD>']] * (max_conv_len - len(X_seq[ci]))
                Y_seq[ci] = Y_seq[ci] + [da_vocab.word2id['<PAD>']] * (max_conv_len - len(Y_seq[ci]))

            assert len(X_seq) == len(Y_seq), 'Unexpect sequence length'

            for i in range(0, max_conv_len):
                X_tensor = torch.tensor([[X[i]] for X in X_seq]).to(device)
                Y_tensor = torch.tensor([[Y[i]] for Y in Y_seq]).to(device)
                if config['use_utt']:
                    max_seq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                    # utterance padding
                    for ci in range(len(XU_seq)):
                        XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<UttPAD>']] * (max_seq_len - len(XU_seq[ci][i]))
                        YU_seq[ci][i] = YU_seq[ci][i] + [utt_vocab.word2id['<UttPAD>']] * (max_seq_len - len(YU_seq[ci][i]))
                    XU_tensor = torch.tensor([XU[i] for XU in XU_seq]).to(device)
                    # YU_tensor = torch.tensor([YU[i] for YU in YU_seq]).to(device)
                    YU_tensor = None

                else:
                    XU_tensor, YU_tensor = None, None
                # X_tensor = (batch_size, 1)
                # XU_tensor = (batch_size, 1, seq_len)

                last = True if i == len(X_seq) - 1 else False
    
                if last:
                    loss, context_hidden, utt_context_hidden = model.forward(X_da=X_tensor, Y_da=Y_tensor, X_utt=XU_tensor, Y_utt=YU_tensor,step_size=step_size, 
                                                         da_encoder=da_encoder, da_decoder=da_decoder, da_context=da_context,
                                                         da_context_hidden=context_hidden,
                                                         utt_encoder=utt_encoder, utt_decoder=utt_decoder, utt_context=utt_context,
                                                         utt_context_hidden=utt_context_hidden,
                                                         criterion=criterion, last=last, config=config)
                    print_total_loss += loss
                    plot_total_loss += loss
                    da_encoder_opt.step()
                    da_decoder_opt.step()
                    da_context_opt.step()
                    if config['use_utt']:
                        utt_encoder_opt.step()
                    if config['use_uttcontext']:
                        utt_context_opt.step()

                else:
                    context_hidden, utt_context_hidden = model.forward(X_da=X_tensor, Y_da=Y_tensor, X_utt=XU_tensor, Y_utt=YU_tensor, step_size=step_size, 
                                                   da_encoder=da_encoder, da_decoder=da_decoder, da_context=da_context,
                                                   da_context_hidden=context_hidden,
                                                   utt_encoder=utt_encoder, utt_decoder=utt_decoder, utt_context=utt_context,
                                                   utt_context_hidden=utt_context_hidden,
                                                   criterion=criterion, last=last, config=config)
            k += step_size
        print()
        valid_loss = validation(X_valid=X_valid, Y_valid=Y_valid, XU_valid=XU_valid, YU_valid=YU_valid,
                                model=model, da_encoder=da_encoder, da_decoder=da_decoder, da_context=da_context,
                                utt_encoder=utt_encoder, utt_context=utt_context, utt_decoder=utt_decoder, config=config)

        if _valid_loss is None:
            _valid_loss = valid_loss
        else:
            if _valid_loss > valid_loss:
                torch.save(da_encoder.state_dict(), os.path.join(config['log_dir'], 'enc_beststate.model'))
                torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_beststate.model'))
                torch.save(da_context.state_dict(), os.path.join(config['log_dir'], 'context_beststate.model'))
                if config['use_utt']:
                    torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_beststate.model'))
                if config['use_uttcontext']:
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
            torch.save(da_encoder.state_dict(), os.path.join(config['log_dir'], 'enc_state{}.model'.format(e + 1)))
            torch.save(da_decoder.state_dict(), os.path.join(config['log_dir'], 'dec_state{}.model'.format(e + 1)))
            torch.save(da_context.state_dict(), os.path.join(config['log_dir'], 'context_state{}.model'.format(e + 1)))
            if config['use_utt']:
                torch.save(utt_encoder.state_dict(), os.path.join(config['log_dir'], 'utt_enc_state{}.model'.format(e + 1)))
            if config['use_uttcontext']:
                torch.save(utt_context.state_dict(), os.path.join(config['log_dir'], 'utt_context_state{}.model'.format(e + 1)))


    print()
    print('Finish training | exec time: %.4f [sec]' % (time.time() - start))

def validation(X_valid, Y_valid, XU_valid, YU_valid, model,
               da_encoder, da_decoder, da_context,
               utt_encoder, utt_context, utt_decoder, config):

    da_context_hidden = da_context.initHidden(1, device)
    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    k = 0

    for seq_idx in range(len(X_valid)):
        X_seq = X_valid[seq_idx]
        Y_seq = Y_valid[seq_idx]
        if config['use_utt']:
            XU_seq = XU_valid[seq_idx]
            YU_seq = YU_valid[seq_idx]

        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in evaluate'

        for i in range(0, len(X_seq)):
            X_tensor = torch.tensor([[X_seq[i]]]).to(device)
            Y_tensor = torch.tensor([[Y_seq[i]]]).to(device)
            if config['use_utt']:
                XU_tensor = torch.tensor([XU_seq[i]]).to(device)
                YU_tensor = torch.tensor([YU_seq[i]]).to(device)
            else:
                XU_tensor, YU_tensor = None, None

            loss, context_hidden, utt_context_hidden = model.evaluate(X_da=X_tensor, Y_da=Y_tensor, X_utt=XU_tensor, Y_utt=YU_tensor,
                                                  da_encoder=da_encoder, da_decoder=da_decoder, da_context=da_context,
                                                  da_context_hidden=da_context_hidden,
                                                  utt_encoder=utt_encoder, utt_decoder=utt_decoder, utt_context=utt_context,
                                                  utt_context_hidden=utt_context_hidden,
                                                  criterion=criterion, config=config)
            total_loss += loss
    return total_loss

if __name__ == '__main__':
    train(args.expr)
