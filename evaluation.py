from torch import optim
import time
from utils import *
import random
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--expr', default='DAonly')
parser.add_argument('--gpu', '-g', type=int, default=0, help='input gpu num')
args = parser.parse_args()

def evaluate(experiment):
    print('loading setting "{}"...'.format(experiment))
    config = initialize_env(experiment)
    XD_test, YD_test, XU_test, _, turn_test = create_traindata(config=config, prefix='test')
    da_vocab = da_Vocab(config, create_vocab=False)
    utt_vocab = utt_Vocab(config, create_vocab=False)
    XD_test = da_vocab.tokenize(XD_test)
    YD_test = da_vocab.tokenize(YD_test)
    XU_test = utt_vocab.tokenize(XU_test)
    predictor = DApredictModel(utt_vocab=utt_vocab, da_vocab=da_vocab, config=config)
    predictor.load_state_dict(torch.load(os.path.join(config['log_dir'], 'da_pred_statevalidbest.model'), map_location=lambda storage, loc: storage))
    batch_size = config['BATCH_SIZE']
    k = 0
    indexes = [i for i in range(len(XU_test))]
    acc = []
    macro_f = []
    while k < len(indexes):
        step_size = min(batch_size, len(indexes) - k)
        batch_idx = indexes[k: k + step_size]
        XU_seq = [XU_test[seq_idx] for seq_idx in batch_idx]
        XD_seq = [XD_test[seq_idx] for seq_idx in batch_idx]
        YD_seq = [YD_test[seq_idx] for seq_idx in batch_idx]
        turn_seq = [turn_test[seq_idx] for seq_idx in batch_idx]
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
        preds = predictor.predict(X_da=XD_tensor, X_utt=XU_tensor, turn=turn_tensor, step_size=step_size)
        preds = np.argmax(preds, axis=1)
        acc.append(accuracy_score(y_pred=preds, y_true=YD_tensor.data.tolist()))
        macro_f.append(f1_score(y_true=YD_tensor.data.tolist(), y_pred=preds, average='macro'))
        k += step_size
    print('Avg. Accuracy: ', np.mean(acc))
    print('Avg. macro-F: ', np.mean(macro_f))


if __name__ == '__main__':
    args = parse()
    evaluate(args.expr)
    