import os
import argparse

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel

device = 'cpu'

def init():
    global device

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)
    
    arg_parser.add_argument('--model_file', default='', type=str)

    arg_parser.add_argument('--sample_num', default=100, type=int)
    arg_parser.add_argument('--max_sample_length', default=20, type=int)
    arg_parser.add_argument('--evaluate_ll', action='store_true')
    arg_parser.add_argument('--batch_size', default=32, type=int)

    arg_parser.add_argument('--output_file', default='', type=str)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


def sample(model, sample_num, max_sample_length, batch_size, evaluate_ll=False):
    model.eval()
    
    examples = torch.LongTensor()
    ll_sum, token_num = 0.0, 0
    with torch.no_grad():
        for i in tqdm(range(0, sample_num, batch_size)):
            num_return_seq = min(batch_size, sample_num - i)
            if evaluate_ll:
                res = model.generate(do_sample=True, max_length=max_sample_length+1,
                    num_return_sequences=num_return_seq, top_k=50257, output_scores=True,
                    return_dict_in_generate=True)
            else:
                res = model.generate(do_sample=True, max_length=max_sample_length+1,
                    num_return_sequences=num_return_seq, top_k=50257, output_scores=False,
                    return_dict_in_generate=True)
            
            examples_ = res.sequences.clone().to('cpu')
            examples_ = examples_[:, 1:]

            n, d = examples_.shape
            if d < max_sample_length:
                examples_ = torch.cat((examples_,
                    torch.LongTensor([[model.config.eos_token_id] * (max_sample_length - d)] * n)), dim=1)

            examples = torch.cat((examples, examples_), dim=0)

            # evaluating avg log likelihood:
            if evaluate_ll:
                examples_ = res.sequences.clone().to(device)
                examples_ = examples_[:, 1:]
                scores = torch.stack(res.scores, 1).to(device)
                d = examples_.shape[1]

                mask = torch.ones(num_return_seq, d).type(torch.LongTensor).to(device)
                for j in range(0, num_return_seq):
                    for k in range(0, d):
                        if examples_[j, k] == model.config.eos_token_id:
                            mask[j, k+1:] = 0
                            break

                log_probs = torch.log(torch.softmax(scores, -1))
                log_probs = log_probs[
                    torch.arange(examples_.shape[0]).unsqueeze(-1),
                    torch.arange(examples_.shape[1]).unsqueeze(0),
                    examples_[:,:]]
                log_probs[mask[:,:] == 0] = 0.0

                token_num += torch.sum(torch.sum(mask, dim=-1), dim=-1).item()
                ll_sum += torch.sum(torch.sum(log_probs, -1), -1).item()

    if evaluate_ll:
        ll_per_sample = ll_sum / sample_num
        ll_per_token = ll_sum / token_num
        print(f'll_per_sample: {ll_per_sample}')
        print(f'll_per_token: {ll_per_token}')

    return examples


def write(examples, output_file):
    examples = examples.tolist()
    with open(output_file, 'w') as fout:
        for example in examples:
            fout.write(','.join([str(x) for x in example]) + '\n')


def main():
    args = init()

    print(f'loading {args.model_file} ...')
    model = GPT2LMHeadModel.from_pretrained(args.model_file)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    examples = sample(model, args.sample_num, args.max_sample_length, 
        args.batch_size, evaluate_ll=args.evaluate_ll)

    write(examples, args.output_file)


if __name__ == '__main__':
    main()