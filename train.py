import time
import os
import pyhocon
import torch
from torch import optim
from models import *
from utils import *


if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = 'cuda'
else:
    device = 'cpu'

def initialize_env(name):
    config = pyhocon.ConfigFactory.parse_file('experiments.conf')[name]
    config['log_dir'] = os.path.join(config['log_root'], name)

    return config

def create_data(config):
    posts, cmnts = create_traindata(config)
    X_train, Y_train, X_valid, Y_valid, _, _ = separate_data(posts, cmnts)
    return X_train, Y_train, X_valid, Y_valid

def train():
    print('loading setting "DApredict"...')
    config = initialize_env('DApredict')
    X_train, Y_train, X_valid, Y_valid = create_data(config)
    print('Finish create train data...')
    da_vocab = da_Vocab(config, X_train + X_valid, Y_train + Y_valid)
    print('Finish create vocab dic...')

    # Tokenize sequences
    X_train, Y_train = da_vocab.tokenize(X_train, Y_train)
    X_valid, Y_valid = da_vocab.tokenize(X_valid, Y_valid)

    print('Finish preparing dataset...')

    assert len(X_train) == len(Y_train), 'Unexpect content in train data'
    assert len(X_valid) == len(Y_valid), 'Unexpect content in valid data'

    encoder = DAEncoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'], da_hidden=config['DA_HIDDEN']).to(device)
    decoder = DADecoder(da_input_size=len(da_vocab.word2id), da_embed_size=config['DA_EMBED'], da_hidden=config['DA_HIDDEN']).to(device)
    context = DAContextEncoder(da_hidden=config['DA_HIDDEN']).to(device)
    model = DAestimator().to(device)
    print('Success construct model...')

    lr = config['lr']
    plot_losses = []

    print_total_loss = 0
    plot_total_loss = 0

    encoder_opt = optim.Adam(encoder.parameters(), lr=lr)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr)
    context_opt = optim.Adam(context.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    print('---start training---')

    start = time.time()
    k = 0
    for e in range(config['EPOCH']):

        for seq_idx in range(0, len(X_train)):
            print('\rsequence {}/{} training...'.format(seq_idx+1, len(X_train)), end='')

            X_seq = X_train[seq_idx]
            Y_seq = Y_train[seq_idx]
            assert len(X_seq) == len(Y_seq), 'Unexpect sequence length'
            context_hidden = context.initHidden(device)
            context_opt.zero_grad()
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()

            for i in range(0, len(X_seq)):
                X_tensor = torch.tensor([X_seq[i]]).to(device)
                Y_tensor = torch.tensor([Y_seq[i]]).to(device)
    
                last = True if i + 1 == len(X_train) - 1 else False
    
                if last:
                    loss, context_hidden = model.forward(X_tensor=X_tensor, Y_tensor=Y_tensor,
                                                         encoder=encoder, decoder=decoder, context=context,
                                                         context_hidden=context_hidden,
                                                         criterion=criterion, last=last)
                    print_total_loss += loss
                    plot_total_loss += loss
                    encoder_opt.step()
                    decoder_opt.step()
                    context_opt.step()

                else:
                    context_hidden = model.forward(X_tensor=X_tensor, Y_tensor=Y_tensor,
                                                   encoder=encoder, decoder=decoder, context=context,
                                                   context_hidden=context_hidden,
                                                   criterion=criterion, last=last)

            valid_loss = evaluate(X_valid=X_valid, Y_valid=Y_valid,
                                model=model, encoder=encoder, decoder=decoder, context=context)


            if e + 1 % config['LOGGING_FREQ'] == 0:
                print_loss_avg = print_total_loss / config['LOGGING_FREQ']
                print_total_loss = 0
                print('steps %d\tloss %.4f\tvalid loss %.4f' % (k, print_loss_avg, valid_loss))
                plot_loss_avg = plot_total_loss / config['LOGGING_FREQ']
                plot_losses.append(plot_loss_avg)
                plot_total_loss = 0

            if e + 1 % config['SAVE_MODEL'] == 0:
                print('saving model')
                torch.save(encoder.state_dict(), os.path.join(config['log_dir'], 'enc_state{}.model'.format(e + 1)))
                torch.save(decoder.state_dict(), os.path.join(config['log_dir'], 'dec_state{}.model'.format(e + 1)))
                torch.save(context.state_dict(), os.path.join(config['log_dir'], 'context_state{}.model'.format(e + 1)))


    print('Finish training | exec time: %.4f [sec]' % (start - time.time()))

def evaluate(X_valid, Y_valid, model, encoder, decoder, context):

    context_hidden = context.initHidden(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for seq_idx in range(0, len(X_valid) - 1):
        X_seq = X_valid[seq_idx]
        Y_seq = Y_valid[seq_idx]
        assert len(X_seq) == len(Y_seq), 'Unexpect sequence len in evaluate'

        for i in range(0, len(X_seq)):
            X_tensor = torch.tensor([X_seq[i]]).to(device)
            Y_tensor = torch.tensor([Y_seq[i]]).to(device)
            loss, context_hidden = model.evaluate(X_tensor=X_tensor, Y_tensor=Y_tensor,
                                                 encoder=encoder, decoder=decoder, context=context,
                                                 context_hidden=context_hidden,
                                                 criterion=criterion)
            total_loss += loss
    return total_loss

if __name__ == '__main__':
    train()
