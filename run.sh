#!/bin/bash

# run.sh

# --
# Run on a small example dataset

mkdir -p _results/mutag

find ./data/kdd_datasets/mutag/*gexf -type f |\
    python ./prep.py --label-path ./data/kdd_datasets/mutag.Labels > _results/mutag/wlk

python ./main.py \
    --inpath _results/mutag/wlk \
    --outdir _results/mutag \
    --batch-size 256 \
    --seed 789 \
    --num-fits 10
