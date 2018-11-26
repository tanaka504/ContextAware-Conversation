import os, re, json
import torch

EOS_token = '<EOS>'
BOS_token = '<BOS>'
parallel_pattern = re.compile(r'^(.+?)(\t)(.+?)$')
file_pattern = re.compile(r'^sw\_([0-9]+?)\_([0-9]+?)\.jsonlines$')


class da_Vocab:
    def __init__(self, config, posts, cmnts):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.posts = posts
        self.cmnts = cmnts
        self.construct()

    def construct(self):
        vocab = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2, '<PAD>': 3}
        vocab_count = {}

        for post, cmnt in zip(self.posts, self.cmnts):
            for token in post:
                #token = self.modify(token)
                if token in vocab_count:
                    vocab_count[token] += 1
                else:
                    vocab_count[token] = 1
            for token in cmnt:
                #token = self.modify(token)
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
        X_tensor = [[self.word2id[token] if token in self.word2id else self.word2id['<UNK>'] for token in sentence] for sentence in X_tensor]
        Y_tensor = [[self.word2id[token] if token in self.word2id else self.word2id['<UNK>'] for token in sentence] for sentence in Y_tensor]
        return X_tensor, Y_tensor

    def modify(self, token):
        token = re.sub(r'[\(\)\,\;]', '', token)
        token = token.split('^')[0]
        return token

class utt_Vocab:
    def __init__(self, config, posts, cmnts):
        self.word2id = None
        self.id2word = None
        self.config = config
        self.posts = posts
        self.cmnts = cmnts
        self.construct()

    def construct(self):
        vocab = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2, '<PAD>': 3}
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
    for filename in files:
        with open(os.path.join(config['train_path'], filename), 'r') as f:
            data = f.read().split('\n')
            data.remove('')
            da_seq = []
            utt_seq = []
            for line in data:
                jsondata = json.loads(line)
                da_seq.append(jsondata['DA'][0])
                utt_seq.append(jsondata['sentence'][0].split(' '))
        da_posts.append(da_seq[:-1])
        da_cmnts.append(da_seq[1:])
        utt_posts.append(utt_seq[:-1])
        utt_cmnts.append(utt_seq[1:])
    assert len(da_posts) == len(da_cmnts), 'Unexpect length da_posts and da_cmnts'
    assert len(utt_posts) == len(utt_cmnts), 'Unexpect length utt_posts and utt_cmnts'
    return da_posts, da_cmnts, utt_posts, utt_cmnts


def separate_data(posts, cmnts):
    split_size = round(len(posts) / 10)
    if split_size == 0: split_size = 1
    X_train, Y_train = posts[split_size * 2:], cmnts[split_size * 2:]
    X_valid, Y_valid = posts[split_size: split_size * 2], cmnts[split_size: split_size * 2]
    X_test, Y_test = posts[:split_size], cmnts[:split_size]
    assert len(X_train) == len(Y_train), 'Unexpect to separate train data'
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

if __name__ == '__main__':
    create_traindata()
