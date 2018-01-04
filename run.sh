#!/bin/bash

# run.sh

# --
# Run on example datasets

python ./prep.py \
    --indir ./data/kdd_datasets/mutag \
    --label-path ./data/kdd_datasets/mutag.Labels > mutag.wlk

python ./main.py \
    --inpath mutag.wlk \
    --batch-size 256 \
    --seed 789 \
    --num-fits 10

# --
# More examples -- deprecated API

python ./prep.py \
    --indir ./data/kdd_datasets/ptc \
    --label-path ./data/kdd_datasets/ptc.Labels > ptc.wlk

python ./main.py \
    --inpath mutag.wlk \
    --batch-size 256 \
    --seed 789 \
    --num-fits 10

# python main.py \
#     --indir ./data/kdd_datasets/proteins \
#     --label-path ./data/kdd_datasets/proteins.Labels \
#     --batch-size 512

# python main.py \
#     --indir ./data/kdd_datasets/nci1 \
#     --label-path ./data/kdd_datasets/nci1.Labels \
#     --batch-size 1024

# python main.py \
#     --indir ./data/kdd_datasets/nci109 \
#     --label-path ./data/kdd_datasets/nci109.Labels \
#     --batch-size 1024