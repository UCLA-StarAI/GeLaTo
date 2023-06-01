#!/bin/bash

mkdir -p output

for split in validation test; do

    setting=(
        "0 0 ./models/gpt2-large_unsupervised/checkpoint-1 ./models/hmm_4096_unsupervised/checkpoint-40.weight.th" # unsupervised setting
        "1 0.3 ./models/gpt2-large_supervised/checkpoint-3 ./models/hmm_4096_unsupervised/checkpoint-40.weight.th"  # supervised setting, using unsupervised distilled HMM
        "2 0.3 ./models/gpt2-large_supervised/checkpoint-3 ./models/hmm_4096_supervised/checkpoint-40.weight.th" # supervised setting, using supervised distilled HMM
        )   
    for params in "${setting[@]}"; do
        set -- $params
        seq2seq=$1
        w=$2
        gpt_model_path=$3
        hmm_model_path=$4

    cmd="python src/decode.py --device cuda --cuda_core 0 \
        --hmm_batch_size 128 --seq2seq $seq2seq --w $w \
        --min_sample_length 5 --max_sample_length 32 \
        --num_beams 128 --length_penalty 0.2 \
        --hmm_model_path $hmm_model_path \
        --gpt_model_path $gpt_model_path \
        --dataset_file data/common-gen_$split.json \
        --output_file output/common-gen_${split}_seqseq=${seq2seq}_output.json"

    echo $cmd
    done
done