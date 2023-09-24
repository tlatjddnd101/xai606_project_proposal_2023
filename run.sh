#!/bin/bash

let "gpu=1"
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --alg 'baseline' \
    --num_epoch 20 \
    --seed 0
sleep 2
