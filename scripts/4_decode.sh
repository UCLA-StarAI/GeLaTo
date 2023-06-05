#!/bin/bash

mkdir -p output

# for unsupervised setting 
python src/decode.py --device cuda --cuda_core 0 \
    --hmm_batch_size 128 --seq2seq 0 \
    --min_sample_length 5 --max_sample_length 32 \
    --num_beams 128 --length_penalty 0.2 \
    --hmm_model_path models/hmm_4096_unsupervised/checkpoint-40.weight.th \
    --gpt_model_path models/gpt2-large_unsupervised/checkpoint-1 \
    --dataset_file data/common-gen_validation.json \
    --output_file output/common-gen_validation_unsupervised_output.json


# for supervised setting 
# python src/decode.py --device cuda --cuda_core 0 \
#     --hmm_batch_size 128 --seq2seq 2 --w 0.3 \
#     --min_sample_length 5 --max_sample_length 32 \
#     --num_beams 128 --length_penalty 0.2 \
#     --hmm_model_path models/hmm_4096_supervised/checkpoint-40.weight.th \
#     --gpt_model_path models/gpt2-large_supervised/checkpoint-3 \
#     --dataset_file data/common-gen_validation.json \
#     --output_file output/common-gen_validation_supervised_output.json