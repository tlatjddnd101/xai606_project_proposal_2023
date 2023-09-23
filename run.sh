#!/bin/bash

let "gpu=3"
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --num_epoch=10 \
    --seed 0
sleep 2
