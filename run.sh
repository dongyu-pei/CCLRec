#!/bin/bash

# ===============================
# Run script for anonymous review
# ===============================

DATASET=ml-1m
GPU_ID=0
SEED=42

python main.py \
    --dataset ${DATASET} \
    --gpu_id ${GPU_ID} \
    --seed ${SEED} \
    --epoch 1000 \
    --batch_size 1024 \
    --embed_size 64 \
    --lr 0.001 \
    --tau 0.1
