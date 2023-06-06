#!/bin/bash

hidden_states=4096
cuda_core=0

# for unsupervised setting 
mkdir -p models/hmm_${hidden_states}_unsupervised

julia --project src/hmm_train.jl --cuda_id $cuda_core \
    --model_path models/hmm_${hidden_states}_unsupervised/ \
    --checkpoint 0 --max_epochs 40 --sample_length 32 \
    --hidden_states $hidden_states --vocab_size 50257 --batch_size 2048 \
    --pseudocount 0.1 \
    --log_file logs/3_train_hmm_unsupervised.log \
    --train_data_file data/unsupervised/common-gen.train


# for supervised setting
# julia --project src/hmm_train.jl --cuda_id $cuda_core \
#     --model_path models/hmm_${hidden_states}_supervised/ \
#     --checkpoint 0 --max_epochs 40 --sample_length 32 \
#     --hidden_states $hidden_states --vocab_size 50257 --batch_size 2048 \
#     --pseudocount 0.1 \
#     --log_file logs/3_train_hmm_supervised.log \
#     --train_data_file data/supervised/common-gen.train