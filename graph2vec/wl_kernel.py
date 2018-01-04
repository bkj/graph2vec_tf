#!/usr/bin/env python

"""
    wl_kernel.py
"""

from __future__ import print_function

import pickle
import networkx as nx
from hashlib import md5

def safe_hash(x):
    return md5(pickle.dumps(x)).hexdigest()

def wl_kernel(graph_file, label_field, wl_height):
    
    # Initialize WL
    g = nx.read_gexf(graph_file)
    
    for node in g.nodes():
        label = g.node[node].get(label_field, 0)
        g.node[node]['relabel'] = {0: safe_hash(label)}
    
    # Apply WL kernel at multiple heights
    for height in range(1, wl_height + 1):
        for node in g.nodes():
            neib_label = tuple(sorted([g.nodes[neib]['relabel'][height - 1] for neib in nx.all_neighbors(g, node)]))
            label = (g.nodes[node]['relabel'][height - 1],) + neib_label
            g.node[node]['relabel'].update({height : safe_hash(label)})
    
    # Write to file
    outpath = graph_file + '.wlk'
    with open(outpath, 'w') as outfile:
        for n, d in g.nodes(data=True):
            for height in range(0, wl_height + 1):
                print('%d+%s' % (height, d['relabel'][height]), file=outfile)
    
    return outpath