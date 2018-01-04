#!/usr/bin/env python

"""
    prep.py
""" 

from __future__ import print_function

import os
import pickle
import argparse
from glob import glob
import networkx as nx
from hashlib import md5

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser("graph2vec")
    
    parser.add_argument("--indir", default = "./data/kdd_datasets/ptc")
    parser.add_argument("--label-path", default='./data/kdd_datasets/ptc.Labels')
    parser.add_argument("--label-field", default='Label')
    parser.add_argument("--wl-height", default=3, type=int)
    
    args = parser.parse_args()
    
    assert os.path.exists(args.indir), "%s does not exist" % args.indir
    
    return args

# --
# Helpers

def get_class_labels(graph_files, label_path):
    graph_to_class_label_map = {l.split()[0].split('.')[0] : int(l.split()[1].strip()) for l in open(label_path)}
    class_labels = [graph_to_class_label_map[os.path.basename(g).split('.')[0]] for g in graph_files]
    return dict(zip(graph_files, class_labels))


def safe_hash(x):
    return md5(pickle.dumps(x)).hexdigest()

# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    
    graph_files = sorted(glob(os.path.join(args.indir, '*.gexf')))
    class_labels = get_class_labels(graph_files, args.label_path)
    
    for graph_file in graph_files:
        class_label = class_labels[graph_file]
        
        # Load graph
        g = nx.read_gexf(graph_file)
        
        # Init WL kernel
        for node in g.nodes():
            label = g.node[node].get(args.label_field, 0)
            g.node[node]['relabel'] = {0: safe_hash(label)}
        
        # Apply WL kernel at multiple heights
        for height in range(1, args.wl_height + 1):
            for node in g.nodes():
                neib_label = tuple(sorted([g.nodes[neib]['relabel'][height - 1] for neib in nx.all_neighbors(g, node)]))
                label = (g.nodes[node]['relabel'][height - 1],) + neib_label
                g.node[node]['relabel'].update({height : safe_hash(label)})
        
        # Write to file
        for n, d in g.nodes(data=True):
            for height, relabel in d['relabel'].items():
                print('%s\t%d\t%d+%s' % (graph_file, class_label, height, relabel))

