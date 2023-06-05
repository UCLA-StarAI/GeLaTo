import os
import json
import argparse

from tqdm import tqdm
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from lemminflect import getAllInflections

from hmm_model import *

device = 'cuda'

class GPTConstraintModel(GPT2LMHeadModel):
    def forward(self, **kwargs):
        input_ids = kwargs['input_ids']

        hmm_model = kwargs['hmm_model']
        hmm_cnf = kwargs['hmm_cnf']
        hmm_seq_len = kwargs['hmm_seq_len']
        hmm_prompt_len = kwargs['hmm_prompt_len']
        hmm_seq2seq = kwargs['hmm_seq2seq']
        hmm_w = kwargs['hmm_w']
        gpt_only = kwargs['gpt_only']
        hmm_only = kwargs['hmm_only']
        hmm_fix_order = kwargs['hmm_fix_order']

        prefixes = [tuple(prefix) for prefix in input_ids[:,1:].tolist()]

        hmm_logits_alpha, hmm_logits = hmm_model.compute_logits(
            prefixes, hmm_cnf, hmm_seq_len, hmm_prompt_len,
            hmm_seq2seq, fix_order=hmm_fix_order)

        kwargs_ = kwargs.copy()
        del kwargs_['hmm_model']
        del kwargs_['hmm_cnf']
        del kwargs_['hmm_seq_len']
        del kwargs_['hmm_prompt_len']
        del kwargs_['hmm_seq2seq']
        del kwargs_['hmm_w']
        del kwargs_['hmm_only']
        del kwargs_['gpt_only']
        del kwargs_['hmm_fix_order']

        outputs = super().forward(**kwargs_)

        hmm_logits_alpha = torch.log_softmax(hmm_logits_alpha, dim=-1)
        hmm_logits = torch.log_softmax(hmm_logits, dim=-1)
        gpt_logits = torch.log_softmax(outputs.logits[:,-1,:], dim=-1)

        if hmm_only:
            logits_new = hmm_logits_alpha
        elif gpt_only:
            logits_new = gpt_logits
        else:
            if hmm_seq2seq:
                logits_new = hmm_w * hmm_logits_alpha + (1.0 - hmm_w) * gpt_logits
            else:
                logits_new = hmm_logits_alpha + gpt_logits - hmm_logits

        logits_new = torch.log_softmax(logits_new, dim=-1)

        outputs.logits[:,-1,:] = logits_new

        return outputs


    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, **model_kwargs)

        inputs['hmm_model'] = model_kwargs['hmm_model']
        inputs['hmm_cnf'] = model_kwargs['hmm_cnf']
        inputs['hmm_seq_len'] = model_kwargs['hmm_seq_len']
        inputs['hmm_prompt_len'] = model_kwargs['hmm_prompt_len']
        inputs['hmm_seq2seq'] = model_kwargs['hmm_seq2seq']
        inputs['hmm_w'] = model_kwargs['hmm_w']
        inputs['gpt_only'] = model_kwargs['gpt_only']
        inputs['hmm_only'] = model_kwargs['hmm_only']
        inputs['hmm_fix_order'] = model_kwargs['hmm_fix_order']

        return inputs


def init():
    global device
    global CUDA_CORE

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='1', type=str)
    arg_parser.add_argument('--hmm_batch_size', default=256, type=int)

    arg_parser.add_argument('--seq2seq', default=0, type=int)
    # --seq2seq 0: unsupervised setting, use the unsupervised base model together with the 
    #              HMM distilled from the unsupervised base model.
    # --seq2seq 1: supervised setting 1, use the supervised base model together with the HMM
    #              model distilled from the unsuperivised base model    
    # --seq2seq 2: supervised setting 2, use supervised base model and the HMM distilled from
    #              the supervised base model
    arg_parser.add_argument('--w', default=0.2, type=float) 
    # weight for geometric mean, only effective with --seq2seq non-zero
    arg_parser.add_argument('--hmm_only', action='store_true')
    arg_parser.add_argument('--gpt_only', action='store_true')    

    arg_parser.add_argument('--min_sample_length', default=5, type=int)
    arg_parser.add_argument('--max_sample_length', default=32, type=int)
    arg_parser.add_argument('--num_beams', default=2, type=int)
    arg_parser.add_argument('--length_penalty', default=0.2, type=float)
    arg_parser.add_argument('--fix_order', action='store_true')
    arg_parser.add_argument('--no_inflection', action='store_true')

    arg_parser.add_argument('--hmm_model_path', default=None, type=str)
    arg_parser.add_argument('--gpt_model_path', default='gpt2', type=str)
    arg_parser.add_argument('--dataset_file', default='', type=str)
    arg_parser.add_argument('--dataset_start', default=0, type=int)
    arg_parser.add_argument('--dataset_end', default=-1, type=int)
    arg_parser.add_argument('--output_file', default='pred.json', type=str)

    args = arg_parser.parse_args()

    # device = f'cuda:{args.cuda_core}' # args.device
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core
    torch.cuda.set_device(int(args.cuda_core))

    return args


def concepts2cnf(concepts, tokenizer, no_inflection=False):
    cnf = []
    concept_set = set([tuple(tokenizer.encode(f' {x}')) for x in concepts])

    for concept in concepts:
        s = tuple(tokenizer.encode(f' {concept}'))
        inflections = set([s])
        if not no_inflection:
            for k, v in getAllInflections(concept).items():
                for x in v:
                    t = tuple(tokenizer.encode(f' {x}'))
                    if len(s) <= len(t) and t[:len(s)] == s:
                        continue
                    # when both surf and surfer are required concepts
                    # avoid the case that surfer is considered an inflection of surf
                    if t in concept_set:
                        continue
                    inflections.add(t)

        clause = tuple(inflections)
        cnf.append(clause)

    cnf = tuple(cnf)

    return cnf


def load_dataset(dataset_file, dataset_start=0, dataset_end=-1):
    with open(dataset_file, 'r') as fin:
        examples = json.load(fin)
    if dataset_end == -1:
        dataset_end = len(examples)-1
        
    examples_ = {}
    for example in examples:
        idx = example['concept_set_idx']
        if dataset_start <= idx and idx <= dataset_end:
            examples_[idx] = {
                'concept_set_idx': idx,
                'concepts': example['concepts'],
                'sentences': [],
            }
    
    examples = [v for _, v in examples_.items()]
    
    return examples


def main():
    args = init()

    print(f'loading gpt2 from {args.gpt_model_path} ...')
    gpt_model = GPTConstraintModel.from_pretrained(args.gpt_model_path)
    gpt_model.config.pad_token_id = gpt_model.config.eos_token_id
    gpt_model.config.use_cache = False
    gpt_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    # pre-define sep_tokens
    sep_tokens = []
    for token in range(0, 50257):
        char = tokenizer.decode(token)[0]
        if char in [' ', '.', ',',]:
            sep_tokens.append(token)

    print(f'loading hmm from {args.hmm_model_path} ...')
    hmm_model = HMM(args.hmm_model_path, sep_tokens=sep_tokens)
    hmm_model.to(device)

    examples = load_dataset(args.dataset_file, args.dataset_start, args.dataset_end)

    print('generating sequences ...')
    counter = 0
    for example_idx in tqdm(range(0, len(examples))):
        example = examples[example_idx]
        concepts = example['concepts']

        cnf = concepts2cnf(concepts, tokenizer, no_inflection=args.no_inflection)

        prompt = '<|endoftext|>'
        if args.seq2seq:
            prompt += ' ' + ' '.join(concepts) + ' ='
        prompt = tuple(tokenizer.encode(prompt))

        if args.seq2seq == 0 or args.seq2seq == 1:
            hmm_seq_len = args.max_sample_length
        else:
            hmm_seq_len = len(prompt) - 1 + args.max_sample_length

        if args.seq2seq == 0:
            hmm_prompt_len = 0
        else:
            hmm_prompt_len = len(prompt) - 1

        model_kwargs = {
            'hmm_model': hmm_model,
            'hmm_cnf': cnf,
            'hmm_seq_len': hmm_seq_len,
            'hmm_prompt_len': hmm_prompt_len,
            'hmm_seq2seq': args.seq2seq,
            'hmm_w': args.w,
            'gpt_only': args.gpt_only,
            'hmm_only': args.hmm_only,
            'hmm_fix_order': args.fix_order
        }

        input_ids = torch.tensor([prompt], device=device)
        with torch.no_grad():
            hmm_model.initialize_cache(hmm_seq_len, cnf,
                prompt_tokens=prompt[1:], batch_size=args.hmm_batch_size, fix_order=args.fix_order)

            outputs = gpt_model.generate(input_ids=input_ids, do_sample=False,
                        num_beams=args.num_beams, num_return_sequences=args.num_beams,
                        min_length=len(prompt)+args.min_sample_length, max_length=len(prompt)+args.max_sample_length,
                        top_k=50257, length_penalty=args.length_penalty, no_repeat_ngram_size=4,
                        output_scores=False, return_dict_in_generate=True, **model_kwargs)

        output_ids = outputs.sequences.detach()

        sentences = [x.strip() for x in tokenizer.batch_decode(
            output_ids[:,len(prompt):], 
            skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        examples[example_idx]['sentences'] = sentences

        with open(args.output_file, 'w') as fout:
            json.dump(examples[:example_idx+1], fout, indent=2)


if __name__ == '__main__':
    main()
