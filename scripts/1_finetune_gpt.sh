#!/bin/bash

mkdir -p logs

# for unsupervised setting 

python src/gpt_finetune.py --device cuda --cuda_core 0 \
    --max_epoch 5 --batch_size 128  --lr 1e-6 \
    --train_data_file ./data/common-gen_train.json \
    --validation_data_file ./data/common-gen_validation.json \
    --model_path ./models/gpt2-large_unsupervised/ \
    --log_file ./logs/1_finetune_gpt_unsupervised.log


# for supervised setting 
# python src/gpt_finetune.py --device cuda --cuda_core 0 \
#     --seq2seq \
#     --max_epoch 5 --batch_size 128  --lr 1e-6 \
#     --train_data_file ./data/common-gen_train.json \
#     --validation_data_file ./data/common-gen_validation.json \
#     --model_path ./models/gpt2-large_supervised/ \
#     --log_file ./logs/1_finetune_gpt_supervised.log