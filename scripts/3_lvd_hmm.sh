#!/bin/bash

hidden_states=4096
cuda_core=0

# for unsupervised setting 
mkdir -p models/hmm_${hidden_states}_unsupervised

python src/hmm_lvd.py --device cuda --cuda_core $cuda_core \
    --teacher_model_checkpoint ./models/gpt2-large_unsupervised/checkpoint-1 \
    --sample_num 500000 --max_sample_length 32 --batch_size 512 \
    --hidden_states $hidden_states --vocab_size 50257 --kmeans_iterations 200 --pseudocount 0.001 \
    --output_file models/hmm_${hidden_states}_unsupervised/checkpoint-0.weight

# for supervised setting
# mkdir -p models/hmm_${hidden_states}_supervised

# python src/hmm_lvd.py --device cuda --cuda_core $cuda_core \
#     --teacher_model_checkpoint ./models/gpt2-large_supervised/checkpoint-3 \
#     --sample_num 500000 --max_sample_length 32 --batch_size 512 \
#     --hidden_states $hidden_states --vocab_size 50257 --kmeans_iterations 200 --pseudocount 0.001 \
#     --output_file models/hmm_${hidden_states}_supervised/checkpoint-0.weight