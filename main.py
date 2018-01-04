#!/usr/bin/env python

"""
    main.py
""" 

from __future__ import print_function

import os
import sys
import argparse
import numpy as np
from glob import glob
import networkx as nx

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV

from graph2vec.make_graph2vec_corpus import *
from graph2vec.corpus_parser import Corpus
from graph2vec.skipgram import Skipgram

# --
# Helpers

def get_class_labels(graph_files, label_path):
    graph_to_class_label_map = {l.split()[0].split('.')[0] : int(l.split()[1].strip()) for l in open(label_path)}
    return np.array([graph_to_class_label_map[os.path.basename(g).split('.')[0]] for g in graph_files])

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser("graph2vec")
    
    parser.add_argument("--indir", default = "./data/kdd_datasets/ptc")
    parser.add_argument("--label-path", default='./data/kdd_datasets/ptc.Labels')
    parser.add_argument("--label-field", default='Label')
    
    parser.add_argument("--embedding-dim", default=512, type=int)
    parser.add_argument("--wl-height", default=3, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.5, type=float)
    parser.add_argument("--num-negsample", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument('--num-fits', type=int, default=1)
    
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    
    assert os.path.exists(args.indir), "%s does not exist" % args.indir
    
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    np.random.seed(args.seed)
    
    # --
    # IO
    
    graph_files = sorted(glob(os.path.join(args.indir, '*.gexf')))
    
    label_lookup = {}
    graphs = [nx.read_gexf(graph_file) for graph_file in graph_files]
    graphs = [initial_relabel(g, args.label_field, label_lookup) for g in graphs]
    
    for height in range(1, args.wl_height + 1):
        label_lookup = {}
        graphs = [wl_relabel(graph, height, label_lookup) for graph in graphs]
    
    for graph_file, graph in zip(graph_files, graphs):
        dump_sg2vec_str(graph_file, args.wl_height, graph)
    
    wlk_files = sorted(glob(os.path.join(args.indir, '*.g2v' + str(args.wl_height))))
    
    # --
    # Featurize graphs
    
    corpus = Corpus(wlk_files)
    
    skipgram_model = Skipgram(
        corpus=corpus,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        num_negsample=args.num_negsample,
    )
    
    X = skipgram_model.train(
        corpus=corpus,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    y = get_class_labels(wlk_files, args.label_path)
    
    # np.save('.X', X)
    # X = np.load('.X.npy')
    
    # --
    # Train classifier
    
    for _ in range(args.num_fits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=np.random.randint(10000))
        svc = GridSearchCV(LinearSVC(), {'C' : 10.0 ** np.arange(-2, 4)}, cv=5, scoring='f1', verbose=0)
        svc.fit(X_train, y_train)
        print("acc=%f" % metrics.accuracy_score(y_test, svc.predict(X_test)))

