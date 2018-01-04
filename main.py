#!/usr/bin/env python

"""
    main.py
""" 

from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV

from graph2vec import Corpus, Skipgram

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser("graph2vec")
    
    parser.add_argument("--inpath", default="./mutag.wlk")
    
    parser.add_argument("--embedding-dim", default=512, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.5, type=float)
    parser.add_argument("--num-negsample", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument('--num-fits', type=int, default=1)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    np.random.seed(args.seed)
    
    # --
    # IO
    
    df = pd.read_csv(args.inpath, sep='\t', header=None)
    df.columns = ('graph', 'class_label', 'subgraph')
    
    corpus = Corpus(df)
    
    # # --
    # # Train model
    
    # skipgram_model = Skipgram(
    #     num_graphs=len(corpus.graph_lookup),
    #     num_subgraphs=len(corpus.subgraph_lookup),
    #     subgraph_frequencies=corpus.subgraph_frequencies,
    #     lr=args.lr,
    #     embedding_dim=args.embedding_dim,
    #     num_negsample=args.num_negsample,
    # )
    
    # graph_features = skipgram_model.train(
    #     corpus=corpus,
    #     num_epochs=args.epochs,
    #     batch_size=args.batch_size,
    # )
    
    # np.save('graph_features', graph_features)
    graph_features = np.load('graph_features.npy')
    # --
    # Train classifier (multiple times, to get idea of variance)
    
    for _ in range(args.num_fits):
        X_train, X_test, y_train, y_test = train_test_split(graph_features, corpus.y, test_size=0.1, random_state=np.random.randint(10000))
        svc = GridSearchCV(LinearSVC(), {'C' : 10.0 ** np.arange(-2, 4)}, cv=5, scoring='f1', verbose=0)
        svc.fit(X_train, y_train)
        print("acc=%f" % metrics.accuracy_score(y_test, svc.predict(X_test)))

