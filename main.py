#!/usr/bin/env python

"""
    main.py
""" 

from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from time import time

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV

from graph2vec.utils import get_files
from graph2vec.make_graph2vec_corpus import *
from graph2vec.corpus_parser import Corpus
from graph2vec.skipgram import Skipgram

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser("graph2vec")
    
    parser.add_argument("--indir", default = "./data/kdd_datasets/ptc")
    parser.add_argument("--label-path", default='./data/kdd_datasets/ptc.Labels')
    parser.add_argument("--label-field", default='Label')
    
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--embedding-dim", default=1024, type=int)
    parser.add_argument("--num-negsample", default=10, type=int)
    parser.add_argument("--lr", default=0.3, type=float)
    parser.add_argument("--wl-height", default=3, type=int)
    
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    
    assert os.path.exists(args.indir), "%s does not exist" % args.indir
    
    return args


def get_class_labels(graph_files, label_path):
    graph_to_class_label_map = {l.split()[0].split('.')[0] : int(l.split()[1].strip()) for l in open(label_path)}
    return np.array([graph_to_class_label_map[os.path.basename(g).split('.')[0]] for g in graph_files])

# --
# Run

args = parse_args()

np.random.seed(args.seed)

# >>
args.indir = './data/kdd_datasets/mutag'
args.label_path = './data/kdd_datasets/mutag.Labels'
args.embedding_dim = 512
args.lr = 0.5
# <<

# --
# IO

graph_files = get_files(dirname=args.indir, extn='.gexf', max_files=0)
# wlk_relabel_and_dump_memory_version(graph_files, max_h=args.wl_height, node_label_attr_name=args.label_field)
wlk_files = get_files(dirname=args.indir, extn='g2v' + str(args.wl_height))

# --
# Train skipgram model

corpus = Corpus(args.indir, wlk_files)

skipgram_model = Skipgram(
    corpus=corpus,
    lr=args.lr,
    embedding_dim=args.embedding_dim,
    num_negsample=args.num_negsample,
    seed=args.seed,
)

X = skipgram_model.train(
    corpus=corpus,
    num_epochs=args.epochs,
    batch_size=args.batch_size,
)

# --
# Train classifier

y = get_class_labels(wlk_files, args.label_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed)

classifier = GridSearchCV(LinearSVC(), {'C' : 10.0 ** np.arange(-2, 4)}, cv=5, scoring='f1', verbose=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(metrics.classification_report(y_test, y_pred), file=sys.stderr)

