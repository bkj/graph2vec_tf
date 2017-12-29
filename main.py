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
    args = argparse.ArgumentParser("graph2vec")
    args.add_argument("-c","--corpus", default = "./data/kdd_datasets/ptc",
        help="Path to directory containing graph files to be used for graph classification or clustering")
    
    args.add_argument('-l','--class_labels_file_name', default='./data/kdd_datasets/ptc.Labels',
        help='File name containg the name of the sample and the class labels')
    
    args.add_argument('-o', "--output_dir", default = "./embeddings",
        help="Path to directory for storing output embeddings")
    
    args.add_argument('-b',"--batch_size", default=128, type=int,
        help="Number of samples per training batch")
    
    args.add_argument('-e',"--epochs", default=1000, type=int,
        help="Number of iterations the whole dataset of graphs is traversed")
    
    args.add_argument('-d',"--embedding_size", default=1024, type=int,
        help="Intended graph embedding size to be learnt")
    
    args.add_argument('-neg', "--num_negsample", default=10, type=int,
        help="Number of negative samples to be used for training")
    
    args.add_argument('-lr', "--learning_rate", default=0.3, type=float,
        help="Learning rate to optimize the loss function")
    
    args.add_argument("--wlk_h", default=3, type=int, 
        help="Height of WL kernel (i.e., degree of rooted subgraph features to be considered for representation learning)")
    
    args.add_argument('-lf', '--label_filed_name', default='Label', 
        help='Label field to be used for coloring nodes in graphs using WL kenrel')
    
    args.add_argument('--seed', type=int, default=123)
    
    return args.parse_args()


def get_class_labels(graph_files, class_labels_file_name):
    graph_to_class_label_map = {l.split()[0].split('.')[0] : int(l.split()[1].strip()) for l in open(class_labels_file_name)}
    return np.array([graph_to_class_label_map[os.path.basename(g).split('.')[0]] for g in graph_files])

# --
# Run

args = parse_args()

# >>
args.corpus = './data/kdd_datasets/mutag'
args.class_labels_file_name = './data/kdd_datasets/mutag.Labels'
args.embedding_size = 512
args.learning_rate = 0.5
# <<

assert os.path.exists(args.corpus), "File {} does not exist".format(args.corpus)
assert os.path.exists(args.output_dir), "Dir {} does not exist".format(args.output_dir)

graph_files = get_files(dirname=args.corpus, extn='.gexf', max_files=0)

wlk_relabel_and_dump_memory_version(graph_files, max_h=args.wlk_h, node_label_attr_name=args.label_filed_name)

wl_extn = 'g2v' + str(args.wlk_h)
wlk_files = get_files(dirname=args.corpus, extn=wl_extn, max_files=0)

# --
# Train skipgram model

embedding_fname = os.path.join(args.output_dir, '_'.join([
    os.path.basename(args.corpus), 
    'dims',   str(args.embedding_size),
    'epochs', str(args.epochs), 
    'lr',     str(args.learning_rate),
    'embeddings.txt'
]))

corpus = Corpus(args.corpus, extn=wl_extn, max_files=0)
corpus.scan_and_load_corpus()

# --
# Train classifier

X = Skipgram(
    corpus=corpus,
    
    learning_rate=args.learning_rate,
    embedding_size=args.embedding_size,
    num_negsample=args.num_negsample,
    num_epochs=args.epochs,
    batch_size=args.batch_size
).train()

y = get_class_labels(wlk_files, args.class_labels_file_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed)

classifier = GridSearchCV(LinearSVC(), {'C' : 10.0 ** np.arange(-2, 4)}, cv=5, scoring='f1', verbose=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(metrics.classification_report(y_test, y_pred), file=sys.stderr)

