
#DONE: imports
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())
from collections import OrderedDict
import os
import sys
if "../models" not in sys.path: # only add this once
    sys.path.append("../models") # add the Visualization package to python_path so we can use it here
if "../evaluation/word-embeddings-benchmarks/" not in sys.path: # only add this once
    sys.path.append("../evaluation/word-embeddings-benchmarks/") # add the Visualization package to python_path so we can use it here
print(sys.path)
import pickle
import collections
from pathlib import Path
import pandas as pd
from web.evaluate import evaluate_similarity, evaluate_on_all_no_logs
from collections import defaultdict
import copy
from web.embedding import Embedding
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#DONE: embedding loader
class LazyLoadEmbeddings():
    """ Class that loads embeddings and the according word list lazyly with file
        path.vec as key. Expected input format is fasttext .vec file - i.e.,
        'word Emb_dim1 dim2 ... dimN'

        Useful, if for a task multiple embedding models (fasttext, glove etc.)
        will be tried out """

    def __init__(self):
        self.loaded_embeddings = dict()

    def load_vec_file(self, path='../training_data/wiki_1B_model.ft.vec'):
        if path not in self.loaded_embeddings:
            print('read_embeddings', path)
            ft_vocab = list()
            ft_embeddings = list()
            with open(path, 'r', encoding="utf-8") as vecs:
                num_words, emb_dim = next(vecs).split() # skip count header
                print("loading", num_words, "words at dim", emb_dim)
                self.expected_tokens_per_line = int(emb_dim) + 1
                for vec in vecs:
                    data = vec.strip().split(' ')
                    if len(data) == self.expected_tokens_per_line:
                        ft_vocab.append(data[0]) # special case for broken data
                        ft_embeddings.append(data[1:])
            print(len(ft_vocab), 'words in fasttext vocab')
            self.loaded_embeddings[path] = dict(embs=ft_embeddings, vocab=ft_vocab)
        return self.loaded_embeddings[path]['embs'], self.loaded_embeddings[path]['vocab']

def give_words_batch(ft_embeddings, batch_size, nptype=np.float32):
    batch_num = int(len(ft_embeddings) / batch_size)
    for X_b in np.array_split(np.array(ft_embeddings, dtype=nptype), batch_num):
        yield X_b

def linear(x):
    return x

class L1Penalty(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input, None

#DONE: MoRTy models
class SparseAutoEncoder(torch.nn.Module):

    def __init__(self, params):
        super(SparseAutoEncoder_NO_BIAS, self).__init__()
        emb_dim = params['emb_d']
        hidden_size = params['rep_dim'] # default: same as original
        self.lin_encoder = nn.Linear(emb_dim, hidden_size)
        self.lin_decoder = nn.Linear(hidden_size, emb_dim)
        self.feature_size = emb_dim
        self.hidden_size = hidden_size
        # l1 for simple sparse AE, had little effect on SUM performance.
        self.l1weight = params['l1']
        self.act = params['activation'] # linear was best for SUM score

    def forward(self, input):
        # encoder
        r = self.act(self.lin_encoder(input))
        # decoder
        if self.l1weight is not None:
            # sparsity penalty
            x_ = self.lin_decoder(L1Penalty.apply(r, self.l1weight))
        else:
            # no sparsity penalty
            x_ = self.lin_decoder(r)
        return x_, r

#HACK: Morty main helper functions #############################################
def relu1(x): # did not work
    # relu that restricts activations to 0-1
    return x.clamp(min=0, max=1)

def MSE(pred, gold, reduce=torch.mean): # worked, but slightly worse than RMSE
    return reduce(((pred - gold) ** 2).mean(0))

def RMSE(pred, gold, reduce=torch.mean): # works, but slower than MSE
    return torch.sqrt(MSE(pred, gold, reduce))

def get_AE_representations(embeddings, device, ptdtype, npdtype, model, batch_size=1000):
    # get model embeddings
    emb_list = [] # to use little memory, we batch
    for i in range(0, len(embeddings), batch_size): # to save memory
        embs = embeddings[i:i+batch_size]
        e = torch.tensor(np.asarray(embs, dtype=npdtype), requires_grad=False, device=device, dtype=ptdtype)
        a , representations = model.forward(e)
        emb_list.append(representations.cpu().data.numpy())
    return np.concatenate(emb_list)

def train_dev_split(embs, vocab, train_size_fraq=.9):
    df = pd.DataFrame()
    df['embs'] = embs
    df['vocab'] = vocab
    msk = np.random.rand(len(df)) < train_size_fraq
    train = df[msk]
    dev = df[~msk]
    return train['embs'].values.tolist(), train['vocab'].values.tolist(), dev['embs'].values.tolist(), dev['vocab'].values.tolist()

def get_optimizer(hyper_conf, net):
    return hyper_conf['optim']['obj'](filter(lambda x: x.requires_grad, net.parameters()), # filter params that are non-tuneable (e.g. Embedding layer)
                                      lr=hyper_conf['lr'],
                                      **hyper_conf['optim']['params'])

#HACK: train loop
## parameter setting
pc = OrderedDict([('model', [SparseAutoEncoder]),
                  ('embs', ['../training_data/ft-crawl-300d-2M.vec',
                            ]),
                   ('vocab_size', ['added_on_the_fly']),
                   ('batch_size', [128]),
                   ('activation', [linear]), # relu1, F.sigmoid
                   ('epochs', [1]),
                   ('train_frac', [.999]), # to see hold out loss -- not needed
                   ('l1', [None]), # 0.05, 0.001, 0.5
                   ('reduce', [torch.mean]), # torch.sum, torch.mean,
                   ('emb_d', ['added_on_the_fly']), # added by code
                   ('loss', [RMSE]), # MSE
                   ('rep_dim', [300, 600]), # overcomplete is less stable
                   # {"obj":torch.optim.SGD, "params":{"momentum":0.9}}]),
                   ('optim', [{"obj":torch.optim.Adam, "params":{"eps":1e-08}}]),
                   ('lr', [2e-02]) #  same as original (if annealed then ~1/2)
                ])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ptdtype = torch.float
    if ptdtype == torch.float:
        npdtype = np.float32

deque = collections.deque(5))) # used for a 5-moving-average loss
for batch in give_words_batch(train_embs, bs):
    x_b = torch.tensor(batch, requires_grad=False, device=device, dtype=ptdtype)
    pred, _ = model.forward(x_b)
    loss = pc['loss'](pred, x_b, pc['reduce'])
    loss.backward()
    optimizer.step() # update parameters
    optimizer.zero_grad()
    ls = float(loss.cpu().data.numpy())
    deque.append(ls)
    print(np.mean(deque), pc['loss'] ) # should fall, but not to 0
#TODO: get the representations
representations = get_AE_representations(embs, device, ptdtype, npdtype, model)
new_embeddings = dict(zip(vocab, representations))

#TODO: evlauate results
res = {k:v[0] for k, v in evaluate_on_all_no_logs(new_embeddings).to_dict().items()}
print(res)
