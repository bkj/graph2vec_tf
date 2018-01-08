#!/usr/bin/env python

"""
    prep.py
""" 

from __future__ import print_function

import os
import sys
import json
import pickle
import argparse
import pandas as pd
from glob import glob
import networkx as nx
from hashlib import md5

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser("graph2vec")
    parser.add_argument("--graph-format", type=str, default='gexf')
    parser.add_argument("--label-path", type=str, default='./_data/mutag/labels')
    parser.add_argument("--label-field", type=str, default='Label')
    parser.add_argument("--wl-height", type=int, default=3)
    return parser.parse_args()

# --
# Helpers

def safe_hash(x):
    return md5(pickle.dumps(x)).hexdigest()

# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    
    class_labels = pd.read_csv(args.label_path, sep=' ', header=None)
    class_labels = dict(zip(class_labels[0], class_labels[1]))
    
    for graph_file in sys.stdin:
        graph_file = graph_file.strip()
        class_label = class_labels[os.path.basename(graph_file)]
        
        # Load graph
        if args.graph_format == 'gexf':
            g = nx.read_gexf(graph_file)
            for node in g.nodes():
                label = g.node[node].get(args.label_field, 0)
                g.node[node]['relabel'] = {0: safe_hash(label)}
        
        elif args.graph_format == 'edgelist':
            g = nx.read_edgelist(graph_file, delimiter='\t', data=(('weight',float),))
            for node in g.nodes:
                g.node[node]['relabel'] = {0: safe_hash(node)}
        else:
            raise Exception('prep.py: unknown format')
        
        # Apply WL kernel at multiple heights
        for height in range(1, args.wl_height + 1):
            for node in g.nodes():
                neib_label = tuple(sorted([g.nodes[neib]['relabel'][height - 1] for neib in nx.all_neighbors(g, node)]))
                label = (g.nodes[node]['relabel'][height - 1],) + neib_label
                g.node[node]['relabel'].update({height : safe_hash(label)})
        
        # Write to file
        wlk = []
        for n, d in g.nodes(data=True):
            for height, relabel in d['relabel'].items():
                wlk.append('%d+%s' % (height, relabel))
        
        print(json.dumps({
            "graph_file" : graph_file,
            "class_label" : class_label,
            "subgraphs" : wlk,
        }))

