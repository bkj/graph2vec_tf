#!/usr/bin/env python

"""
    make_graph2vec_corpus.py
"""

import networkx as nx

def get_int_node_label(l):
    return int(l.split('+')[-1])


def initial_relabel(g, label_field, label_lookup):
    
    nx.convert_node_labels_to_integers(g, first_label=0)
    for node in g.nodes():
        g.node[node]['relabel'] = {}
    
    for node in g.nodes():
        try:
            label = g.node[node][label_field]
        except:
            g.node[node]['relabel'][0] = '0+0'
            continue
            
        if not label_lookup.has_key(label):
            compressed_label = len(label_lookup) + 1 #starts with 1 and incremented every time a new node label is seen
            label_lookup[label] = compressed_label #inster the new label to the label map
            g.node[node]['relabel'][0] = '0+' + str(compressed_label)
        else:
            g.node[node]['relabel'][0] = '0+' + str(label_lookup[label])
    
    return g


def wl_relabel(g, height, label_lookup):
    
    prev_height = height - 1
    for node in g.nodes():
        prev_height_node_label = get_int_node_label(g.nodes[node]['relabel'][prev_height])
        node_label = [prev_height_node_label]
        neighbors = list(nx.all_neighbors(g, node))
        neighborhood_label = sorted([get_int_node_label(g.nodes[nei]['relabel'][prev_height]) for nei in neighbors])
        node_neighborhood_label = tuple(node_label + neighborhood_label)
        if not label_lookup.has_key(node_neighborhood_label):
            compressed_label = len(label_lookup) + 1
            label_lookup[node_neighborhood_label] = compressed_label
            g.node[node]['relabel'][height] = str(height) + '+' + str(compressed_label)
        else:
            g.node[node]['relabel'][height] = str(height) + '+' + str(label_lookup[node_neighborhood_label])
    
    return g


def dump_sg2vec_str(fname, wl_height, g):
    
    opfname = fname + '.g2v' + str(wl_height)
    
    with open(opfname,'w') as fh:
        for n,d in g.nodes(data=True):
            for i in xrange(0, wl_height+1):
                try:
                    center = d['relabel'][i]
                except:
                    continue
                
                neis_labels_prev_deg = []
                neis_labels_next_deg = []
                
                if i != 0:
                    neis_labels_prev_deg = [g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g, n)]
                    neis_labels_prev_deg = sorted(list(set(neis_labels_prev_deg)))
                
                neis_labels_same_deg = list(set([g.node[nei]['relabel'][i] for nei in nx.all_neighbors(g,n)]))
                
                if i != wl_height:
                    neis_labels_next_deg = [g.node[nei]['relabel'][i+1] for nei in nx.all_neighbors(g,n)]
                    neis_labels_next_deg = sorted(list(set(neis_labels_next_deg)))
                    
                nei_list = neis_labels_same_deg + neis_labels_prev_deg + neis_labels_next_deg
                nei_list = ' '.join(nei_list)
                
                sentence = center + ' ' + nei_list
                # print >> fh, sentence
                print(opfname, sentence)
