#!/bin/bash

cuda_core=0

# for unsupervised setting 
mkdir -p logs

LOG_FILE="./logs/1_finetune_gpt_unsupervised.log"

cmd="python src/gpt_finetune.py --device cuda --cuda_core $cuda_core \
    --max_epoch 5 --batch_size 128  --lr 1e-6 \
    --train_data_file ./data/common-gen_train.json \
    --validation_data_file ./data/common-gen_validation.json \
    --model_path ./models/gpt2-large_unsupervised/ \
    --log_file $LOG_FILE"

echo $cmd >> $LOG_FILE

$cmd

# for supervised setting 

LOG_FILE="./logs/1_finetune_gpt_supervised.log"

cmd="python src/gpt_finetune.py --device cuda --cuda_core $cuda_core \
    --seq2seq \
    --max_epoch 5 --batch_size 128  --lr 1e-6 \
    --train_data_file ./data/common-gen_train.json \
    --validation_data_file ./data/common-gen_validation.json \
    --model_path ./models/gpt2-large_supervised/ \
    --log_file $LOG_FILE"

echo $cmd >> $LOG_FILE

$cmd