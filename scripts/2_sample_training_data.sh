#!/bin/bash

cuda_core=0

# for unsupervised setting 
mkdir -p ./data/unsupervised/
for idx in {1..40}
do
    python src/gpt_sample_data.py --device cuda --cuda_core $cuda_core \
        --model_file ./models/gpt2-large_unsupervised/checkpoint-1 \
        --sample_num 200000 --max_sample_length 32 --batch_size 1024 \
        --output_file ./data/unsupervised/common-gen.train.${idx}
done

# for supervised setting
# mkdir -p ./data/supervised/
# for idx in {1..40}
# do
#     python src/gpt_sample_data.py --device cuda --cuda_core $cuda_core \
#         --model_file ./models/gpt2-large_supervised/checkpoint-3 \
#         --sample_num 200000 --max_sample_length 32 --batch_size 1024 \
#         --output_file ./data/supervised/common-gen.train.${idx}
# done