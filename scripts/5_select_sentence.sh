#!/bin/bash

cuda_core=0
gpt_model_path=models/gpt2-large_unsupervised/checkpoint-1 # always use `unsupervised` gpt2-large

for input_file in output/*_output.json; do
    output_file=$(echo $input_file | cut -d'.' -f 1)
    output_file=${output_file}_selected.json
    cmd="python src/select_sentence.py --device cuda --cuda_core $cuda_core \
        --rerank --gpt_model_path $gpt_model_path \
        --input_file $input_file \
        --output_file $output_file"
    echo $cmd
done 