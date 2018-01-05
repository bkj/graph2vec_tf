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
