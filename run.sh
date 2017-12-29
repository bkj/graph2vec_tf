#!/bin/bash

# run.sh

tar -xzvf data.tar.gz && rm data.tar.gz

cd src
python main.py --corpus ./data/kdd_datasets/mutag --class_labels_file_name ./data/kdd_datasets/mutag.Labels 

python main.py --corpus ./data/kdd_datasets/proteins --class_labels_file_name ./data/kdd_datasets/proteins.Labels \
    --batch_size 16 --embedding_size 128 --num_negsample 5