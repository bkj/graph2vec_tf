
from __future__ import print_function

import sys
import math
import tensorflow as tf

class Skipgram(object):
    def __init__(self, corpus, learning_rate, embedding_size, num_negsample, num_epochs, batch_size):
        
        self.corpus         = corpus
        self.embedding_size = embedding_size
        self.num_negsample  = num_negsample
        self.learning_rate  = learning_rate
        self.num_epochs     = num_epochs
        self.batch_size     = batch_size
        
        # --
        # Define network
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.batch_inputs = tf.placeholder(tf.int32, shape=([None, ]))
            self.batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))
            
            graph_embeddings = tf.Variable(tf.random_uniform([self.corpus.num_graphs, self.embedding_size], -0.5 / self.embedding_size, 0.5/self.embedding_size))
            batch_graph_embeddings = tf.nn.embedding_lookup(graph_embeddings, self.batch_inputs)
            weights = tf.Variable(tf.truncated_normal([self.corpus.num_subgraphs, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
            biases = tf.Variable(tf.zeros(self.corpus.num_subgraphs))
            
            #negative sampling part
            self.loss = tf.reduce_mean(
              tf.nn.nce_loss(
                weights=weights,
                biases=biases,
                labels=self.batch_labels,
                inputs=batch_graph_embeddings,
                num_sampled=self.num_negsample,
                num_classes=self.corpus.num_subgraphs,
                sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                  true_classes=self.batch_labels,
                  num_true=1,
                  num_sampled=self.num_negsample,
                  unique=True,
                  range_max=self.corpus.num_subgraphs,
                  distortion=0.75,
                  unigrams=self.corpus.subgraph_id_freq_map_as_list)
              )
            )
            
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100000, 0.96, staircase=True) #linear decay over time
            learning_rate = tf.maximum(learning_rate,0.001) #cannot go below 0.001 to ensure at least a minimal learning
            
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
            self.normalized_embeddings = graph_embeddings / tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims=True))
    
    def train(self):
        tf_config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)
        with tf.Session(graph=self.graph, config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(self.num_epochs):
                loss, step = 0, 0
                while self.corpus.epoch_flag == False:
                    batch_data, batch_labels = self.corpus.generate_batch_from_file(self.batch_size)
                    
                    _, loss_val = sess.run([self.optimizer,self.loss], feed_dict={
                        self.batch_inputs: batch_data,
                        self.batch_labels: batch_labels
                    })
                    
                    loss += loss_val
                    step += 1
                
                print('Skipgram: epoch: %d | avg_loss: %f' % (i, loss / step), file=sys.stderr)
                self.corpus.epoch_flag = False
            
            final_embeddings = self.normalized_embeddings.eval()
          
        return final_embeddings
