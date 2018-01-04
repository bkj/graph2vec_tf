#!/usr/bin/env python

"""
    graph2vec.py
"""

from __future__ import print_function

import sys
import math
import numpy as np
import tensorflow as tf
from collections import Counter

class Corpus(object):
    def __init__(self, df):
        class_lookup = dict(zip(df.graph, df.class_label))
        
        # Graph names to unique IDs
        self.y = class_lookup.values()
        self.graph_lookup = {g:i for i, g in enumerate(class_lookup.keys())}
        self.graph_ids = np.array(df.graph.apply(lambda x: self.graph_lookup[x]))
        
        # Subgraphs to unique IDs, ordered by frequency
        subgraph2freq = Counter(list(df.subgraph) + ["UNK"])
        self.subgraph_frequencies = subgraph2freq.values()
        self.subgraph_lookup = {sg:i for i, sg in enumerate(subgraph2freq.keys())}
        self.subgraph_ids = np.array(df.subgraph.apply(lambda x: self.subgraph_lookup[x])).reshape(-1, 1)
    
    def iterate(self, batch_size):
        idx = np.random.permutation(self.graph_ids.shape[0])
        for chunk in np.array_split(idx, idx.shape[0] / batch_size):
            yield self.graph_ids[chunk], self.subgraph_ids[chunk]


class Skipgram(object):
    def __init__(self, num_graphs, num_subgraphs, subgraph_frequencies, lr, embedding_dim, num_negsample):
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            
            self.graph_ids = tf.placeholder(tf.int32, shape=([None, ]))
            self.subgraph_ids = tf.placeholder(tf.int64, shape=([None, 1]))
            
            graph_embeddings = tf.Variable(tf.random_uniform([num_graphs, embedding_dim], -0.5 / embedding_dim, 0.5/embedding_dim, seed=np.random.randint(10000)))
            batch_graph_embeddings = tf.nn.embedding_lookup(graph_embeddings, self.graph_ids)
            weights = tf.Variable(tf.truncated_normal([num_subgraphs, embedding_dim], stddev=1.0 / math.sqrt(embedding_dim), seed=np.random.randint(10000)))
            biases = tf.Variable(tf.zeros(num_subgraphs))
            
            self.loss = tf.reduce_mean(
              tf.nn.nce_loss(
                weights=weights,
                biases=biases,
                labels=self.subgraph_ids,
                inputs=batch_graph_embeddings,
                num_sampled=num_negsample,
                num_classes=num_subgraphs,
                sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                  true_classes=self.subgraph_ids,
                  num_true=1,
                  num_sampled=num_negsample,
                  unique=True,
                  range_max=num_subgraphs,
                  distortion=0.75,
                  unigrams=subgraph_frequencies,
                  seed=np.random.randint(10000),
                )
              )
            )
            
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(lr, global_step, 100000, 0.96, staircase=True)
            lr = tf.maximum(lr, 0.001)
            
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=global_step)
            self.ngraph_embeddings = graph_embeddings / tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims=True))
    
    def train(self, corpus, num_epochs, batch_size):
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list="0"
        with tf.Session(graph=self.graph, config=config) as sess:
            
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(num_epochs):
                loss, step = 0, 0
                for graph_ids, subgraph_ids in corpus.iterate(batch_size):
                    _, loss_val = sess.run([self.optimizer, self.loss], feed_dict={
                        self.graph_ids: graph_ids,
                        self.subgraph_ids: subgraph_ids
                    })
                    
                    loss += loss_val
                    step += 1
                
                print('Skipgram: epoch: %d | avg_loss: %f' % (epoch, loss / step), file=sys.stderr)
            
            return self.ngraph_embeddings.eval()
