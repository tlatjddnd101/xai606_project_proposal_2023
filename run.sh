#!/bin/bash

let "gpu=0"
XLA_PYTHON_CLIENT_MEM_FRACTION=.1 CUDA_VISIBLE_DEVICES=$gpu python main.py \
    --config=config.py \
    --epoch=1000000 \
    --seed 77
sleep 2
