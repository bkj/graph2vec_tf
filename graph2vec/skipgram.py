#!/usr/bin/env python

"""
    skipgram.py
"""

from __future__ import print_function

import sys
import math
import numpy as np
import tensorflow as tf

class Skipgram(object):
    def __init__(self, corpus, lr, embedding_dim, num_negsample):
        
        # --
        # Define graph
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            self.batch_inputs = tf.placeholder(tf.int32, shape=([None, ]))
            self.batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))
            
            graph_embeddings = tf.Variable(tf.random_uniform([corpus.num_graphs, embedding_dim], -0.5 / embedding_dim, 0.5/embedding_dim, seed=np.random.randint(10000)))
            batch_graph_embeddings = tf.nn.embedding_lookup(graph_embeddings, self.batch_inputs)
            weights = tf.Variable(tf.truncated_normal([corpus.num_subgraphs, embedding_dim], stddev=1.0 / math.sqrt(embedding_dim), seed=np.random.randint(10000)))
            biases = tf.Variable(tf.zeros(corpus.num_subgraphs))
            
            self.loss = tf.reduce_mean(
              tf.nn.nce_loss(
                weights=weights,
                biases=biases,
                labels=self.batch_labels,
                inputs=batch_graph_embeddings,
                num_sampled=num_negsample,
                num_classes=corpus.num_subgraphs,
                sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                  true_classes=self.batch_labels,
                  num_true=1,
                  num_sampled=num_negsample,
                  unique=True,
                  range_max=corpus.num_subgraphs,
                  distortion=0.75,
                  unigrams=corpus.subgraph_id_freq_map_as_list,
                  seed=np.random.randint(10000),
                )
              )
            )
            
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(lr, global_step, 100000, 0.96, staircase=True)
            lr = tf.maximum(lr, 0.001)
            
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=global_step)
            self.normalized_embeddings = graph_embeddings / tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims=True))
    
    def train(self, corpus, num_epochs, batch_size):
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=False)) as sess:
            
            sess.run(tf.global_variables_initializer())
            
            for i in range(num_epochs):
                loss, step = 0, 0
                while not corpus.epoch_flag:
                    batch_data, batch_labels = corpus.generate_batch_from_file(batch_size)
                    
                    _, loss_val = sess.run([self.optimizer, self.loss], feed_dict={
                        self.batch_inputs: batch_data,
                        self.batch_labels: batch_labels
                    })
                    
                    loss += loss_val
                    step += 1
                
                print('Skipgram: epoch: %d | avg_loss: %f' % (i, loss / step), file=sys.stderr)
                corpus.epoch_flag = False
            
            final_embeddings = self.normalized_embeddings.eval()
          
        return final_embeddings
