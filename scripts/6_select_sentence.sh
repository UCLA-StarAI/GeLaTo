#!/bin/bash

# for unsupervised setting 
python src/select_sentence.py --device cuda --cuda_core 0 \
        --rerank --gpt_model_path models/gpt2-large_unsupervised/checkpoint-1 \
        --input_file output/common-gen_validation_unsupervised_output.json \
        --output_file output/common-gen_validation_unsupervised_selected.json


# for supervised setting 
# python src/select_sentence.py --device cuda --cuda_core 0 \
#         --rerank --gpt_model_path models/gpt2-large_unsupervised/checkpoint-1 \
#         --input_file output/common-gen_validation_supervised_output.json \
#         --output_file output/common-gen_validation_supervised_selected.json