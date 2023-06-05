#!/bin/bash

# for unsupervised setting 
python src/eval.py \
    --result_file common-gen_validation_unsupervised_selected.json \
    --target_file common-gen_validation.json 

# for supervised setting 
# python src/eval.py \
#     --result_file common-gen_validation_supervised_selected.json \
#     --target_file common-gen_validation.json 