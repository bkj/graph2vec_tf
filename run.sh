#!/bin/bash

# run.sh

# --
# Run on a small example dataset

mkdir -p _results/mutag

find ./_data/mutag/graphs -type f |\
    python ./prep.py --label-path ./_data/mutag/labels > _results/mutag/wlk.jl

python ./main.py \
    --inpath _results/mutag/wlk.jl \
    --outdir _results/mutag \
    --batch-size 256 \
    --seed 789 \
    --num-fits 10

# --
# Run on malware dataset

mkdir -p _results/malware/class

# wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v1/graph2vec/malware/class.tar.gz
# tar -xzf class.tar.gz && rm class.tar.gz
# mv class _data/malware/class

# Compute WL kernels
find ./_data/malware/class/{all2013,allnewbenign} -type f |\
    parallel --pipe -N 100 "python ./prep.py --label-path ./_data/malware/class/labels --graph-format edgelist" > _results/malware/class/wlk.jl

# Remove infrequent WL kernel tokens
python ./filter.py --inpath _results/malware/class/wlk.jl --outpath _results/malware/class/wlk.filtered.jl

# Run graph2vec
python ./main.py \
    --inpath _results/malware/class/wlk.filtered.jl \
    --outdir _results/malware/class \
    --batch-size 128 \
    --seed 789 \
    --num-fits 10 \
    --epochs 50
