import os, re, json
import matplotlib.pyplot as plt
import torch

EOS_token = '<EOS>'
BOS_token = '<BOS>'
parallel_pattern = re.compile(r'^(.+?)(\t)(.+?)$')
file_pattern = re.compile(r'^sw\_([0-9]+?)\_([0-9]+?)\.jsonlines$')

damsl_align = {'<Uninterpretable>': ['%', 'x'],
               '<Statement>': ['sd', 'sv', '^2', 'no', 't3', 't1', 'oo', 'cc', 'co', 'oo_co_cc'],
               '<Question>': ['q', 'qy', 'qw', 'qy^d', 'bh', 'qo', 'qh', 'br', 'qrr', '^g', 'qw^d'],
               '<Directive>': ['ad'],
               '<Propose>': ['p'],
               '<Greeting>': ['fp', 'fc'],
               '<Apology>': ['fa', 'nn', 'ar', 'ng', 'nn^e', 'arp', 'nd', 'arp_nd'],
               '<Agreement>': ['aa', 'aap', 'am', 'aap_am', 'ft'],
               '<Understanding>': ['b', 'bf', 'ba', 'bk', 'na', 'ny', 'ny^e'],
               '<Other>': ['o', 'fo', 'bc', 'by', 'fw', 'h', '^q', 'b^m', '^h', 'bd', 'fo_o_fw_"_by_bc'],
               '<turn>': ['<turn>']}

class da_Vocab:
    def __init__(self, config, posts, cmnts):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.posts = posts
        self.cmnts = cmnts
        self.construct()

    def construct(self):
        vocab = {'<PAD>': 0}
        vocab_count = {}

        for post, cmnt in zip(self.posts, self.cmnts):
            for token in post:
                if token in vocab_count:
                    vocab_count[token] += 1
                else:
                    vocab_count[token] = 1
            for token in cmnt:
                if token in vocab_count:
                    vocab_count[token] += 1
                else:
                    vocab_count[token] = 1

        for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
            vocab[k] = len(vocab)
            if len(vocab) >= self.config['MAX_VOCAB']: break
        self.word2id = vocab
        self.id2word = {v : k for k, v in vocab.items()}

        return vocab

    def tokenize(self, X_tensor, Y_tensor):
        X_tensor = [[self.word2id[token] for token in sentence] for sentence in X_tensor]
        Y_tensor = [[self.word2id[token] for token in sentence] for sentence in Y_tensor]
        return X_tensor, Y_tensor

class utt_Vocab:
    def __init__(self, config, posts, cmnts):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.posts = posts
        self.cmnts = cmnts
        self.construct()

    def construct(self):
        vocab = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2, '<UttPAD>': 3, '<ConvPAD>': 4}
        vocab_count = {}

        for post, cmnt in zip(self.posts, self.cmnts):
            for seq in post:
                for word in seq:
                    if word in vocab_count:
                        vocab_count[word] += 1
                    else:
                        vocab_count[word] = 1
            for seq in cmnt:
                for word in seq:
                    if word in vocab_count:
                        vocab_count[word] += 1
                    else:
                        vocab_count[word] = 1

        for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
            vocab[k] = len(vocab)
            if len(vocab) >= self.config['UTT_MAX_VOCAB']: break
        self.word2id = vocab
        self.id2word = {v : k for k, v in vocab.items()}

        return vocab

    def tokenize(self, X_tensor, Y_tensor):
        X_tensor = [[[self.word2id[token] if token in self.word2id else self.word2id['<UNK>'] for token in seq] for seq in dialogue] for dialogue in X_tensor]
        Y_tensor = [[[self.word2id[token] if token in self.word2id else self.word2id['<UNK>'] for token in seq] for seq in dialogue] for dialogue in Y_tensor]
        return X_tensor, Y_tensor

def create_traindata(config):
    files = [f for f in os.listdir(config['train_path']) if file_pattern.match(f)]
    da_posts = []
    da_cmnts = []
    utt_posts = []
    utt_cmnts = []
    turn = []
    # 1file 1conversation
    for filename in files:
        with open(os.path.join(config['train_path'], filename), 'r') as f:
            data = f.read().split('\n')
            data.remove('')
            da_seq = []
            utt_seq = []
            turn_seq = []
            # 1line 1turn
            for idx, line in enumerate(data, 1):
                jsondata = json.loads(line)
                # single-turn multi dialogue case
                if config['multi_dialogue']:
                    for da, utt in zip(jsondata['DA'], jsondata['sentence']):
                        da_seq.append(da)
                        utt_seq.append(utt.split(' '))
                        turn_seq.append(0)
                    if not config['turn']:
                        da_seq.append('<turn>')
                        utt_seq.append('<turn>')
                    turn_seq[-1] = 1
                # single-turn single dialogue case
                else:
                    da_seq.append(jsondata['DA'][-1])
                    utt_seq.append(jsondata['sentence'][-1].split(' '))
            da_seq = [easy_damsl(da) for da in da_seq]
            # assert len(turn_seq) == len(da_seq), '{} != {}'.format(len(turn_seq), len(da_seq))
        if config['state']:
            for i in range(max(1, len(da_seq) - 1 - config['window_size'])):
                da_posts.append(da_seq[i:min(len(da_seq)-1, i + config['window_size'])])
                da_cmnts.append(da_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
                utt_posts.append(utt_seq[i:min(len(da_seq)-1, i + config['window_size'])])
                utt_cmnts.append(utt_seq[1 + i:min(len(da_seq), 1 + i + config['window_size'])])
                turn.append(turn_seq[i:min(len(da_seq), i + config['window_size'])])
        else:
            da_posts.append(da_seq[:-1])
            da_cmnts.append(da_seq[1:])
            utt_posts.append(utt_seq[:-1])
            utt_cmnts.append(utt_seq[1:])
            turn.append(turn_seq[:-1])
    assert len(da_posts) == len(da_cmnts), 'Unexpect length da_posts and da_cmnts'
    assert len(utt_posts) == len(utt_cmnts), 'Unexpect length utt_posts and utt_cmnts'
    # assert len(turn) == len(da_posts)
    return da_posts, da_cmnts, utt_posts, utt_cmnts, turn

def easy_damsl(tag):
    easy_tag = [k for k, v in damsl_align.items() if tag in v]
    return easy_tag[0] if not len(easy_tag) < 1 else tag

def separate_data(posts, cmnts, turn):
    split_size = round(len(posts) / 10)
    if split_size == 0: split_size = 1
    X_train, Y_train, Tturn = posts[split_size * 2:], cmnts[split_size * 2:], turn[split_size * 2:]
    X_valid, Y_valid, Vturn = posts[split_size: split_size * 2], cmnts[split_size: split_size * 2], turn[split_size: split_size * 2]
    X_test, Y_test, Testturn = posts[:split_size], cmnts[:split_size], turn[:split_size]
    assert len(X_train) == len(Y_train), 'Unexpect to separate train data'
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, Tturn, Vturn, Testturn

def preprocess(X, mode='X'):
    result_x = []
    result_turn = []
    if mode == 'Y':
        return [[x_seq for x_seq in x_conv if not x_seq == '<turn>'] for x_conv in X], None
    for x_conv in X:
        tmp_x = []
        turn = []
        for x_seq in x_conv:
            if x_seq == '<turn>':
                turn[-1] = 1
            else:
                turn.append(0)
                tmp_x.append(x_seq)
        assert len(tmp_x) == len(turn), '{} | {}'.format(len(tmp_x), len(turn))
        result_x.append(tmp_x)
        result_turn.append(turn)
    return result_x, result_turn

def makefig(X, Y, xlabel, ylabel, imgname):
    plt.figure(figsize=(12, 6))
    plt.bar(X, Y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('./data/images/', imgname))

if __name__ == '__main__':
    create_traindata()
