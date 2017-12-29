import numpy as np
from collections import Counter
from utils import get_files

class Corpus(object):
    def __init__(self, indir, graph_fname_list):
        self.subgraph_index = 0
        self.graph_index = 0
        self.epoch_flag = 0
        self.graph_ids_for_batch_traversal = []
        
        # scan_and_load_corpus
        self.graph_fname_list = graph_fname_list
        self._graph_name_to_id_map = {g: i for i, g in enumerate(self.graph_fname_list)}  # input layer of the skipgram network
        self._id_to_graph_name_map = {i: g for g, i in self._graph_name_to_id_map.iteritems()}
        subgraph_to_id_map = self._scan_corpus()
        self.graph_ids_for_batch_traversal = range(self.num_graphs)
        np.random.shuffle(self.graph_ids_for_batch_traversal)
    
    def _scan_corpus(self):
        
        subgraphs = []
        for fname in self.graph_fname_list:
            subgraphs.extend([l.split()[0] for l in open(fname).xreadlines()])  # just take the first word of every sentence
        
        subgraphs.append('UNK')
        
        subgraph_to_freq_map = Counter(subgraphs)
        del subgraphs
        
        subgraph_to_id_map = {sg: i for i, sg in enumerate(subgraph_to_freq_map.iterkeys())}  # output layer of the skipgram network
        
        self._subgraph_to_freq_map = subgraph_to_freq_map  # to be used for negative sampling
        self._subgraph_to_id_map = subgraph_to_id_map
        self._id_to_subgraph_map = {v:k for k,v in subgraph_to_id_map.iteritems()}
        self._subgraphcount = sum(subgraph_to_freq_map.values()) #total num subgraphs in all graphs
        
        self.num_graphs = len(self.graph_fname_list) #doc size
        self.num_subgraphs = len(subgraph_to_id_map) #vocab of word size
        
        self.subgraph_id_freq_map_as_list = [] #id of this list is the word id and value is the freq of word with corresponding word id
        for i in xrange(len(self._subgraph_to_freq_map)):
            self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])
            
        return self._subgraph_to_id_map
        
    def generate_batch_from_file(self, batch_size):
        target_graph_ids = []
        context_subgraph_ids = []
        
        graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
        graph_contents = open(graph_name).readlines()
        while self.subgraph_index >= len(graph_contents):
            self.subgraph_index = 0
            self.graph_index += 1
            if self.graph_index == len(self.graph_fname_list):
                self.graph_index = 0
                np.random.shuffle(self.graph_ids_for_batch_traversal)
                self.epoch_flag = True
            
            graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
            graph_contents = open(graph_name).readlines()
            
        while len(context_subgraph_ids) < batch_size:
            line_id = self.subgraph_index
            context_subgraph = graph_contents[line_id].split()[0]
            target_graph = graph_name
            
            context_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph])
            target_graph_ids.append(self._graph_name_to_id_map[target_graph])
            
            self.subgraph_index+=1
            while self.subgraph_index == len(graph_contents):
                self.subgraph_index = 0
                self.graph_index += 1
                if self.graph_index == len(self.graph_fname_list):
                    self.graph_index = 0
                    np.random.shuffle(self.graph_ids_for_batch_traversal)
                    self.epoch_flag = True
                    
                graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
                graph_contents = open(graph_name).readlines()
        
        target_context_pairs = zip(target_graph_ids, context_subgraph_ids)
        np.random.shuffle(target_context_pairs)
        target_graph_ids, context_subgraph_ids = zip(*target_context_pairs)
        
        target_graph_ids = np.array(target_graph_ids, dtype=np.int32)
        context_subgraph_ids = np.array(context_subgraph_ids, dtype=np.int32)
        
        contextword_outputs = np.reshape(context_subgraph_ids, [len(context_subgraph_ids), 1])
        
        return target_graph_ids, contextword_outputs