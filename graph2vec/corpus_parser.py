import numpy as np
from collections import Counter

class Corpus(object):
    def __init__(self, wlk_files):
        self.subgraph_index = 0
        self.graph_index    = 0
        self.epoch_flag     = 0
        
        self.graph_ids_for_batch_traversal = []
        
        # scan_and_load_corpus
        self.wlk_files = wlk_files
        self._graph_name_to_id_map = {g: i for i, g in enumerate(self.wlk_files)}  # input layer of the skipgram network
        
        # scan corpus
        subgraphs = []
        for fname in self.wlk_files:
            subgraphs.extend([l.split()[0] for l in open(fname).readlines()])  # just take the first word of every sentence
        
        subgraphs.append('UNK')
        
        subgraph_to_freq_map = Counter(subgraphs)
        del subgraphs
        
        subgraph_to_id_map = {sg: i for i, sg in enumerate(subgraph_to_freq_map.iterkeys())}  # output layer of the skipgram network
        
        self._subgraph_to_freq_map = subgraph_to_freq_map  # to be used for negative sampling
        self._subgraph_to_id_map   = subgraph_to_id_map
        self._id_to_subgraph_map   = {v:k for k,v in subgraph_to_id_map.iteritems()}
        self._subgraphcount        = sum(subgraph_to_freq_map.values()) #total num subgraphs in all graphs
        
        self.num_graphs = len(self.wlk_files) #doc size
        self.num_subgraphs = len(subgraph_to_id_map) #vocab of word size
        
        self.subgraph_id_freq_map_as_list = [] #id of this list is the word id and value is the freq of word with corresponding word id
        for i in xrange(len(self._subgraph_to_freq_map)):
            self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])
        
        self.graph_ids_for_batch_traversal = range(self.num_graphs)
        np.random.shuffle(self.graph_ids_for_batch_traversal)
    
    def generate_batch_from_file(self, batch_size):
        batch_data = []
        batch_labels = []
        
        graph_name = self.wlk_files[self.graph_ids_for_batch_traversal[self.graph_index]]
        graph_contents = open(graph_name).readlines()
        while self.subgraph_index >= len(graph_contents):
            self.subgraph_index = 0
            self.graph_index += 1
            if self.graph_index == len(self.wlk_files):
                self.graph_index = 0
                np.random.shuffle(self.graph_ids_for_batch_traversal)
                self.epoch_flag = True
            
            graph_name = self.wlk_files[self.graph_ids_for_batch_traversal[self.graph_index]]
            graph_contents = open(graph_name).readlines()
            
        while len(batch_labels) < batch_size:
            line_id = self.subgraph_index
            context_subgraph = graph_contents[line_id].split()[0]
            target_graph = graph_name
            
            batch_data.append(self._graph_name_to_id_map[target_graph])
            batch_labels.append(self._subgraph_to_id_map[context_subgraph])
            
            self.subgraph_index += 1
            while self.subgraph_index == len(graph_contents):
                self.subgraph_index = 0
                self.graph_index += 1
                if self.graph_index == len(self.wlk_files):
                    self.graph_index = 0
                    np.random.shuffle(self.graph_ids_for_batch_traversal)
                    self.epoch_flag = True
                    
                graph_name = self.wlk_files[self.graph_ids_for_batch_traversal[self.graph_index]]
                graph_contents = open(graph_name).readlines()
        
        batch_data   = np.array(batch_data, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32).reshape(-1, 1)
        
        idx = np.random.permutation(batch_data.shape[0])
        return batch_data[idx], batch_labels[idx]