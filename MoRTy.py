# %%
#DONE: imports
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.device_count())
from collections import OrderedDict
import os
import sys
if "evaluation/word-embeddings-benchmarks/" not in sys.path: # only add this once
    sys.path.append("evaluation/word-embeddings-benchmarks/") # add the Visualization package to python_path so we can use it here
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
import itertools
if not sys.warnoptions:
    warnings.simplefilter("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#DONE: embedding loader
# %%
class LazyLoadEmbeddings():
    """ Class that loads embeddings and the according word list lazyly with file
        path.vec as key. Expected input format is fasttext .vec file - i.e.,
        'word Emb_dim1 dim2 ... dimN'

        Useful, if for a task multiple embedding models (fasttext, glove etc.)
        will be tried out """

    def __init__(self):
        self.loaded_embeddings = dict()

    def load_vec_file(self, path='../training_data/some_FT_File.ft.vec'):
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

# %%
#DONE: MoRTy models
class SparseAutoEncoder(torch.nn.Module):

    def __init__(self, params):
        super(SparseAutoEncoder, self).__init__()
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

#DONE: Morty main helper functions #############################################
# %%
def relu1(x): # did not work
    # relu that restricts activations to 0-1
    return x.clamp(min=0, max=1)

# %%
def MSE(pred, gold, reduce=torch.mean): # worked, but slightly worse than RMSE
    return reduce(((pred - gold) ** 2).mean(0))

# %%
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
def num_morties_to_create(num_morties, epochs=1, original_embeddings_path="/tmp"):
    """ @param: num_morties how many Reconstructing Transformations to create. 1 epoch = default training.
        @param: epoch, creates one subvariation of the random transformation per epoch
        @param: creates a new path name for every variation
    """
    return [original_embeddings_path + 'epoch:' + str(e) + 'rand:' + str(random_init) for random_init in range(num_morties) for e in range(epochs)]

def filter_on_metric(scores, metric_we_care_about=''):
    #FIXME: define a sensible condition here. E.g. fires when a new max score is reached
    return True # makes no sense atm. == Placeholder

def store_embeddings(path='MoRTy_embedding.', vocab=None, embs=None, store_as='.vec'):
    print(path)
    # vocab and embeddings are list and array, with the same sequence order
    # vocab[2] => embs[2]
    if vocab == None or embs == None:
        raise Exception("empty data dump")
    if store_as == '.pickle':
        with open(path+store_as,'wb') as f:
            pickle.dump({'vocab':vocab, 'embeddings':embs}, f)
    elif store_as == '.vec':
        with open(path+store_as,'w') as f:
            f.write(str(len(embs)) + ' ' + str(len(embs[vocab[0]])) + '\n') # header
            for token, emb in embs.items():
                f.write(token + ' ' + ' '.join(str(x_i) for x_i in emb) + '\n')
    else:
        raise Exception('embedding output format not recognized')

def loaded_embeddings(path='MoRTy_embedding.pkl', as_dict=False):
    """ Return vocab embedding dict. Vocab and and embeddings are mapped by
    order -- i.e. vocab[2] as assigned to embeddings[2]
    @params: as_dict=True returns a {vocab_i:embedding_i, ... } dict """
    with open(path,'rb') as f:
        pickle.load(f)
        if as_dict:
            return dict(zip(f['vocab'], f['embeddings']))
        return pickle.load(f)

def parameter_combos_generator(param_lists_dict):
    """ Use to compare multiple embedders
        Usable for hyperparameter search, but not significantly beneficial.
    """
    return (dict(zip(param_lists_dict, x)) for x in itertools.product(*param_lists_dict.values()))

## parameter setting
def run_MoRTy_to_produce_specialized_embeddings(param_space, evaluate_word_embs=False):
    LLE = LazyLoadEmbeddings() # lazy embedding loader if used for multiple 'embs'

    ptdtype = torch.float
    if ptdtype == torch.float:
        npdtype = np.float32

    # MoRTy ---------------- core of the MoRTy method ''
    for pc in parameter_combos_generator(param_space):
        # load data
        embs, vocab = LLE.load_vec_file(pc['embs'])
        train_embs, train_vocab, dev_embs, dev_vocab = train_dev_split(embs, vocab, train_size_fraq=pc['train_frac'])
        dev_embs = np.array(dev_embs, dtype=npdtype) # string array to float array
        pc['vocab_size'] = len(vocab)
        pc['emb_d'] = len(train_embs[0])

        # train loop
        train_deque = collections.deque([], 5) # used for a 5-moving-average loss
        dev_deque = collections.deque([], 5) # used for a 5-moving-average loss
        model = pc['model'](pc)
        model = model.cuda() if torch.cuda.is_available() else model
        if not torch.cuda.is_available():
            print("WARNING: running on CPUs")
        optimizer = get_optimizer(pc, model) # torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad()
        for RT in num_morties_to_create(num_morties=7, epochs=pc['epochs'], original_embeddings_path=pc['embs']):
            for batch in give_words_batch(train_embs, pc['batch_size']):
                x_b = torch.tensor(batch, requires_grad=False, device=device, dtype=ptdtype)
                pred, _ = model.forward(x_b)
                loss = pc['loss'](pred, x_b, pc['reduce'])
                loss.backward()
                optimizer.step() # update parameters
                optimizer.zero_grad()
                ls = float(loss.cpu().data.numpy())
                train_deque.append(ls)
                dev_deque.append(ls)
                # break
            # DONE: get the representations
            # print final losses
            print(np.mean(train_deque), pc['loss'].__name__, 'on train') # should fall, but not to 0
            print(np.mean(dev_deque), pc['loss'].__name__, 'on') # should fall, but not to 0
            representations = get_AE_representations(embs, device, ptdtype, npdtype, model)
            new_embeddings = dict(zip(vocab, representations))

            #DONE: evlauate results
            if evaluate_word_embs:
                res = {k:v[0] for k, v in evaluate_on_all_no_logs(new_embeddings).to_dict().items()}
                print(res)
            # store the best RT embedding according to a proxy measure OR
            # store k MoRTy to select the optimal RT via a downstream tasks dev set
            if filter_on_metric:
                store_embeddings(path=RT + '.morty', vocab=vocab, embs=new_embeddings)

if __name__ == "__main__":
    # parameter setting or exploration for MoRTy
    pc = OrderedDict([('model', [SparseAutoEncoder]),
                      ('embs', ['data/wikitext2_FastText_SG0.vec'],
                                # 'data/wikitext103_FastText_SG0.vec']# your original embedding
                                ),
                       ('vocab_size', ['added_on_the_fly']),
                       ('batch_size', [128]), #
                       ('activation', [linear]), # relu1, F.sigmoid
                       ('epochs', [4]), # e.g. 5 can 5 new RT embeddings (1 per epoch)
                       ('train_frac', [.999]), # to see hold out/ dev loss -- not needed
                       ('l1', [None]), # 0.05, 0.001, 0.5
                       ('reduce', [torch.mean]), # or torch.sum: did not matter
                       ('emb_d', ['added_on_the_fly']), # added by code
                       ('loss', [RMSE]), # MSE works similarly well (faster)
                       ('rep_dim', [40]), # overcomplete gives some boost
                       # {"obj":torch.optim.SGD, "params":{"momentum":0.9}}])
                       ('optim', [{"obj":torch.optim.Adam, "params":{"eps":1e-08}}]),
                       ('lr', [2e-02]) # ~ same as original (if annealed then ~1/2)
                    ])
    # produce MoRTy versions
    run_MoRTy_to_produce_specialized_embeddings(pc)
