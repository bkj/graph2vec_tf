#!/usr/bin/env python

"""
    main.py
""" 

import os
import argparse
import logging
from time import time

from graph2vec.utils import get_files
from graph2vec.train_utils import train_skipgram
from graph2vec.classify import perform_classification
from graph2vec.make_graph2vec_corpus import *

logger = logging.getLogger()
logger.setLevel("INFO")


def parse_args():
    args = argparse.ArgumentParser("graph2vec")
    args.add_argument("-c","--corpus", default = "../data/kdd_datasets/ptc",
        help="Path to directory containing graph files to be used for graph classification or clustering")
    
    args.add_argument('-l','--class_labels_file_name', default='../data/kdd_datasets/ptc.Labels',
        help='File name containg the name of the sample and the class labels')
    
    args.add_argument('-o', "--output_dir", default = "../embeddings",
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
    
    return args.parse_args()


if __name__=="__main__":
    args = parse_args()
    
    
    wl_extn = 'g2v'+str(args.wlk_h)
    
    assert os.path.exists(args.corpus), "File {} does not exist".format(args.corpus)
    assert os.path.exists(args.output_dir), "Dir {} does not exist".format(args.output_dir)
    
    graph_files = get_files(dirname=args.corpus, extn='.gexf', max_files=0)
    logging.info('Loaded {} graph file names form {}'.format(len(graph_files),args.corpus))
    
    t0 = time()
    wlk_relabel_and_dump_memory_version(graph_files, max_h=args.wlk_h, node_label_attr_name=args.label_filed_name)
    logging.info('dumped sg2vec sentences in {} sec.'.format(time() - t0))
    
    t0 = time()
    embedding_fname = train_skipgram(
        args.corpus,
        wl_extn,
        args.learning_rate,
        args.embedding_size,
        args.num_negsample,
        args.epochs,
        args.batch_size,
        args.output_dir
    )
    
    logging.info('Trained the skipgram model in {} sec.'.format(round(time()-t0, 2)))
    
    perform_classification (args.corpus, wl_extn, embedding_fname, args.class_labels_file_name)
