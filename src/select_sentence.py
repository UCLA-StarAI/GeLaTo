import os
import sys
import json
import argparse

from tqdm import tqdm
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = 'cuda'

def init():
    global device
    global CUDA_CORE

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='1', type=str)

    arg_parser.add_argument('--rerank', action='store_true')
    arg_parser.add_argument('--gpt_model_path', default='gpt2', type=str)
    arg_parser.add_argument('--input_file', default='', type=str)
    arg_parser.add_argument('--output_file', default='', type=str)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


def loglikelihood(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    n, d = input_ids.shape
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:,:-1,:]
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs[torch.arange(0, n).unsqueeze(-1),
            torch.arange(0, d-1).unsqueeze(0), input_ids[:,1:]]
        log_probs *= attention_mask[:,1:]

    lls = log_probs.sum(dim=-1)

    return lls.tolist()


def main():
    args = init()

    if args.rerank:
        print(f'loading gpt2 from {args.gpt_model_path} ...')
        gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_model_path)
        gpt_model.config.pad_token_id = gpt_model.config.eos_token_id
        gpt_model.eval()
        gpt_model.to(device)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        tokenizer.pad_token = tokenizer.eos_token

    examples = []
    processed_examples = []
    with open(args.input_file, 'r') as fin:
        examples = json.load(fin)
        
    for example in tqdm(examples):
        if example['sentences'] != []:
            if args.rerank:
                sentences = ['<|endoftext|>' + x for x in example['sentences']]
                lls = loglikelihood(gpt_model, tokenizer, sentences)
                selected = sorted([(a, b) for a, b in zip(example['sentences'], lls)], 
                    key=lambda x: x[1], reverse=True)[0][0]
            else:
                selected = example['sentences'][0]
        else:
            selected = ''
            continue

        processed_examples.append({
            'concept_set_idx': example['concept_set_idx'],
            'concepts': example['concepts'],
            'sentence': selected,
        })
        
    with open(args.output_file, 'w') as fout:
        json.dump(processed_examples, fout, indent=2)


if __name__ == '__main__':
    main()