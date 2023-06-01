import json
import argparse

import datasets


def load_and_process(output_file, split):
    dataset_raw = datasets.load_dataset('common_gen', split=split)

    examples = []
    for example in dataset_raw:
        idx, concepts, target = example['concept_set_idx'], example['concepts'], example['target']
        examples.append({
            'concept_set_idx': idx,
            'concepts': concepts,
            'target': target
        })

    with open(output_file, 'w') as fout:
        json.dump(examples, fout, indent=2)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output_path', default='hmm.flow', type=str)
    args = arg_parser.parse_args()

    for split in ['train', 'validation', 'test']:
        load_and_process(args.output_path + f'common-gen_{split}.json', split)

if __name__ == '__main__':
    main()