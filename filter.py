#!/usr/bin/env python

"""
    filter.py
"""

from __future__ import print_function

import sys
from tqdm import tqdm
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--min-count', type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    
    print("filter.py: loading graphs", file=sys.stderr)
    graphs = map(json.loads, open(args.inpath))
    
    print("filter.py: counting subgarphs", file=sys.stderr)
    counter = Counter()
    for graph in tqdm(graphs):
        for subgraph in graph['subgraphs']:
            counter[subgraph] += 1
    
    print("filter.py: finding rare tokens", file=sys.stderr)
    keep = set([])
    for k in tqdm(counter):
        if counter[k] >= args.min_count:
            keep.add(k)
    
    print("filter.py: dropping rare tokens", file=sys.stderr)
    for graph in tqdm(graphs):
        graph['subgraphs'] = list(keep.intersection(graph['subgraphs']))
    
    print("filter.py: writing", file=sys.stderr)
    open(args.outpath, 'w').write('\n'.join(map(json.dumps, graphs)))