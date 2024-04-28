#!/bin/bash
python train.py \
    --opt adam \
    --lr 1e-3 \
    --gamma 1e-1 \
    --epoch 200 \
    --stage 1 \
    --val_epoch 10 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 5 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_shot 1 5 \
    --pre \
    --gpu 0
