#!/usr/bin/env python

"""
    make_graph2vec_corpus.py
"""


import os
import networkx as nx

label_to_compressed_label_map = {}

def get_int_node_label(l):
    return int(l.split('+')[-1])

def initial_relabel(g, node_label_attr_name='Label'):
    global label_to_compressed_label_map
    
    nx.convert_node_labels_to_integers(g, first_label=0)  # this needs to be done for the initial interation only
    for node in g.nodes():
        g.node[node]['relabel'] = {}
    
    for node in g.nodes():
        try:
            label = g.node[node][node_label_attr_name]
        except:
            g.node[node]['relabel'][0] = '0+0'
            continue
            
        if not label_to_compressed_label_map.has_key(label):
            compressed_label = len(label_to_compressed_label_map) + 1 #starts with 1 and incremented every time a new node label is seen
            label_to_compressed_label_map[label] = compressed_label #inster the new label to the label map
            g.node[node]['relabel'][0] = '0+' + str(compressed_label)
        else:
            g.node[node]['relabel'][0] = '0+' + str(label_to_compressed_label_map[label])
            
    return g


def wl_relabel(g, it):
    global label_to_compressed_label_map
    
    prev_iter = it - 1
    for node in g.nodes():
        prev_iter_node_label = get_int_node_label(g.nodes[node]['relabel'][prev_iter])
        node_label = [prev_iter_node_label]
        neighbors = list(nx.all_neighbors(g, node))
        neighborhood_label = sorted([get_int_node_label(g.nodes[nei]['relabel'][prev_iter]) for nei in neighbors])
        node_neighborhood_label = tuple(node_label + neighborhood_label)
        if not label_to_compressed_label_map.has_key(node_neighborhood_label):
            compressed_label = len(label_to_compressed_label_map) + 1
            label_to_compressed_label_map[node_neighborhood_label] = compressed_label
            g.node[node]['relabel'][it] = str(it) + '+' + str(compressed_label)
        else:
            g.node[node]['relabel'][it] = str(it) + '+' + str(label_to_compressed_label_map[node_neighborhood_label])
    
    return g


def dump_sg2vec_str(fname, max_h, g):
    
    opfname = fname + '.g2v' + str(max_h)
    
    if os.path.isfile(opfname):
        return
    
    with open(opfname,'w') as fh:
        for n,d in g.nodes(data=True):
            for i in xrange(0, max_h+1):
                try:
                    center = d['relabel'][i]
                except:
                    continue
                
                neis_labels_prev_deg = []
                neis_labels_next_deg = []
                
                if i != 0:
                    neis_labels_prev_deg = list(set([g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g, n)]))
                    neis_labels_prev_deg.sort()
                NeisLabelsSameDeg = list(set([g.node[nei]['relabel'][i] for nei in nx.all_neighbors(g,n)]))
                if i != max_h:
                    neis_labels_next_deg = list(set([g.node[nei]['relabel'][i+1] for nei in nx.all_neighbors(g,n)]))
                    neis_labels_next_deg.sort()
                    
                nei_list = NeisLabelsSameDeg + neis_labels_prev_deg + neis_labels_next_deg
                nei_list = ' '.join(nei_list)
                
                sentence = center + ' ' + nei_list
                print >> fh, sentence
    
    if os.path.isfile(fname+'.tmpg'):
        os.system('rm '+fname+'.tmpg')

def wlk_relabel_and_dump_memory_version(fnames,max_h,node_label_attr_name='Label'):
    global label_to_compressed_label_map
    
    graphs = [nx.read_gexf(fname) for fname in fnames]
    graphs = [initial_relabel(g,node_label_attr_name) for g in graphs]
    
    for it in xrange(1, max_h + 1):
        label_to_compressed_label_map = {}
        graphs = [wl_relabel(g, it) for g in graphs]
    
    for fname, g in zip(fnames, graphs):
        dump_sg2vec_str(fname, max_h, g)
