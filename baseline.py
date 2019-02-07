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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from train import initialize_env, create_DAdata, create_Uttdata, make_batchidx


parser = argparse.ArgumentParser()
parser.add_argument('--expr', '-e', default='baseline', help='input experiment config')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
args = parser.parse_args()

if torch.cuda.is_available():
    # torch.cuda.set_device(args.gpu)
    # device = 'cuda'
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

print('Use device: ', device)

def train(experiment):
    print('loading setting "{}"...'.format(experiment))
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, _, _, Tturn, Vturn, _ = create_DAdata(config)
    print('Finish create train data...')
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    if config['use_utt']:
        XU_train, YU_train, XU_valid, YU_valid, _, _ = create_Uttdata(config)
        utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)
    else:
        utt_vocab = None
    print('Finish create vocab dic...')

    # Y_train, _ = preprocess(Y_train, mode='Y')
    # Y_valid, _ = preprocess(Y_valid, mode='Y')
    # Tokenize sequences
    X_train, Y_train = da_vocab.tokenize(X_train, Y_train)
    X_valid, Y_valid = da_vocab.tokenize(X_valid, Y_valid)
    # XU_train, Tturn = preprocess(XU_train, mode='X')
    # XU_valid, Vturn = preprocess(XU_valid, mode='X')
    XU_train, YU_train = utt_vocab.tokenize(XU_train, YU_train)
    XU_valid, YU_valid = utt_vocab.tokenize(XU_valid, YU_valid)
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
    model = baseline(device).to(device)
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
            max_conv_len = max(len(s) for s in Y_seq)  # seq_len は DA と UTT で共通

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
                turn_tensor = torch.tensor([[t[i]] for t in turn_seq]).float().to(device)
                max_seq_len = max(len(XU[i]) + 1 for XU in XU_seq)
                # utterance padding
                for ci in range(len(XU_seq)):
                    XU_seq[ci][i] = XU_seq[ci][i] + [utt_vocab.word2id['<UttPAD>']] * (max_seq_len - len(XU_seq[ci][i]))
                XU_tensor = torch.tensor([XU[i] for XU in XU_seq]).to(device)
                # YU_tensor = torch.tensor([YU[i] for YU in YU_seq]).to(device)
                # YU_tensor = None

                # X_tensor = (batch_size, 1)
                # XU_tensor = (batch_size, 1, seq_len)

                last = True if i == max_conv_len - 1 else False
    
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
            turn_tensor = torch.tensor([[turn_seq[i]]]).float().to(device)
            XU_tensor = torch.tensor([XU_seq[i]]).to(device)

            loss, utt_context_hidden = model.evaluate(Y_da=Y_tensor, X_utt=XU_tensor,
                                                  turn=turn_tensor, da_decoder=da_decoder,
                                                  utt_encoder=utt_encoder, utt_context=utt_context,
                                                  utt_context_hidden=utt_context_hidden,
                                                  criterion=criterion, config=config)
            total_loss += loss
    return total_loss

def evaluate(experiment):
    print('load vocab')
    config = initialize_env(experiment)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, _, _, turn = create_DAdata(config)
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    XU_train, YU_train, XU_valid, YU_valid, XU_test, YU_test = create_Uttdata(config)
    utt_vocab = utt_Vocab(config, XU_train + XU_valid, YU_train + YU_valid)

    _, Y_test = da_vocab.tokenize(X_test, Y_test)

    # Y_test, _ = preprocess(Y_test, mode='Y')
    XU_test, _ = utt_vocab.tokenize(XU_test, YU_test)
    # XU_test, turn = preprocess(XU_test, mode='X')

    print('load models')
    decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'],
                        da_hidden=config['DEC_HIDDEN']).to(device)

    # loading weight
    decoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'dec_beststate.model')))

    utt_encoder = UtteranceEncoder(utt_input_size=len(utt_vocab.word2id), embed_size=config['UTT_EMBED'], utterance_hidden=config['UTT_HIDDEN'], padding_idx=utt_vocab.word2id['<UttPAD>']).to(device)
    utt_encoder.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_enc_beststate.model')))
    utt_context = UtteranceContextEncoder(utterance_hidden_size=config['UTT_CONTEXT']).to(device)
    utt_context.load_state_dict(torch.load(os.path.join(config['log_dir'], 'utt_context_beststate.model')))

    model = baseline().to(device)

    utt_context_hidden = utt_context.initHidden(1, device) if config['use_uttcontext'] else None

    true = []
    pred = []

    for seq_idx in range(0, len(X_test)):
        print('\r{}/{} conversation evaluating'.format(seq_idx+1, len(Y_test)), end='')
        Y_seq = Y_test[seq_idx]
        turn_seq = turn[seq_idx]
        if config['use_utt']:
            XU_seq = XU_test[seq_idx]

        for i in range(0, len(Y_seq)):
            turn_tensor = torch.tensor([[turn_seq[i]]]).to(device)
            Y_tensor = torch.tensor(Y_seq[i]).to(device)
            XU_tensor = torch.tensor([[XU_seq[i]]]).to(device)

            decoder_output, utt_context_hidden = model.predict(X_utt=XU_tensor, turn=turn_tensor,
                                                               da_decoder=decoder,
                                                               utt_encoder=utt_encoder, utt_context=utt_context,
                                                               utt_context_hidden=utt_context_hidden, config=config)
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
    train(args.expr)
    true, pred, true_detok, pred_detok = evaluate(args.expr)
    calc_average(y_true=true, y_pred=pred)
