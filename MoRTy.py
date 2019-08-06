
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

#HACK: Morty main
#HACK: train loop
