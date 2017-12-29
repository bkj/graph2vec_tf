#!/bin/bash

# run.sh

# --
# Run on example datasets

python main.py \
    --indir ./data/kdd_datasets/mutag \
    --label-path ./data/kdd_datasets/mutag.Labels \
    --batch-size 256 \
    --seed 456

python main.py \
    --indir ./data/kdd_datasets/ptc \
    --label-path ./data/kdd_datasets/ptc.Labels \
    --batch-size 256

python main.py \
    --indir ./data/kdd_datasets/proteins \
    --label-path ./data/kdd_datasets/proteins.Labels \
    --batch-size 512

python main.py \
    --indir ./data/kdd_datasets/nci1 \
    --label-path ./data/kdd_datasets/nci1.Labels \
    --batch-size 1024

python main.py \
    --indir ./data/kdd_datasets/nci109 \
    --label-path ./data/kdd_datasets/nci109.Labels \
    --batch-size 1024