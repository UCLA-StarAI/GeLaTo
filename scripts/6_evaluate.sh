#!/bin/bash

for split in validation test; do
    for seq2seq in 0 1 2; do
        result_file=output/common-gen_${split}_seqseq=${seq2seq}_output_selected
        target_file=data/common-gen_${split}.json
        cmd="python src/eval.py --result_file $result_file.json --target_file $target_file > $result_file.answer"
        echo $cmd
    done
done