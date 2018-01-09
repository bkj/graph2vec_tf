#!/usr/bin/env python

"""
    main.py
""" 

from __future__ import print_function

import os
import sys
import json
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
    
    parser.add_argument("--inpath", default="./_results/malware/class/wlk.filtered.jl")
    parser.add_argument("--outdir", default="./_results/malware/class/")
    
    parser.add_argument("--embedding-dim", default=512, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.5, type=float)
    parser.add_argument("--num-negsample", default=10, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument('--num-fits', type=int, default=1)
    parser.add_argument('--num-jobs', type=int, default=16)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    np.random.seed(args.seed)
    
    # --
    # IO
    
    print("main.py: load graphs", file=sys.stderr)
    graphs = map(json.loads, open(args.inpath))
    corpus = Corpus(graphs)
    
    # --
    # Grab labels
    
    graph_labels = np.array([g['class_label'] for g in graphs])
    np.save(os.path.join(args.outdir, 'labs'), graph_labels)
    
    # --
    # Train model
    
    print("main.py: init skipgram", file=sys.stderr)
    skipgram_model = Skipgram(
        num_graphs=len(corpus.graph_lookup),
        num_subgraphs=len(corpus.subgraph_lookup),
        subgraph_frequencies=corpus.subgraph_frequencies,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        num_negsample=args.num_negsample,
    )
    
    print("main.py: train skipgram", file=sys.stderr)
    ngraph_embeddings = skipgram_model.train(
        corpus=corpus,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    np.save(os.path.join(args.outdir, 'embeddings'), ngraph_embeddings)
    
    # --
    # Train classifier (multiple times, to get an idea of variance)
    
    accs, aucs = [], []
    for _ in range(args.num_fits):
        X_train, X_test, y_train, y_test =\
            train_test_split(ngraph_embeddings, graph_labels, test_size=0.1, random_state=np.random.randint(10000))
        
        svc = GridSearchCV(
            estimator=LinearSVC(),
            param_grid={'C' : 10.0 ** np.arange(-2, 4)},
            cv=5,
            scoring='f1',
            verbose=1,
            n_jobs=args.num_jobs
        )
        svc.fit(X_train, y_train)
        
        test_preds = svc.predict(X_test)
        acc = metrics.accuracy_score(y_test, test_preds)
        auc = metrics.roc_auc_score(y_test, test_preds)
        
        print("acc=%f | auc=%f" % (acc, auc))
        accs.append(acc)
        aucs.append(auc)
    
    print('acc | mean=%f | std=%f' % (np.mean(accs), np.std(accs)))
    print('auc | mean=%f | std=%f' % (np.mean(aucs), np.std(aucs)))

