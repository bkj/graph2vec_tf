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

mkdir -p _results/malware/family
find ./_data/malware/family/ -type f |\
    python ./prep.py --label-path ./_data/malware/labels --graph-format edgelist > _results/malware/family/wlk.jl

python ./main.py \
    --inpath _results/malware/family/wlk.jl \
    --outdir _results/malware/family \
    --batch-size 256 \
    --seed 789 \
    --num-fits 10


mkdir -p _results/malware/package
find ./_data/malware/package/ -type f |\
    python ./prep.py --label-path ./_data/malware/labels --graph-format edgelist > _results/malware/package/wlk.jl
