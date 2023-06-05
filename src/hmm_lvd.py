import os
import argparse

import torch
import numpy
import faiss

from tqdm import tqdm
from transformers import GPT2LMHeadModel

device = 'cuda'

def init():
    global device

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device', default='cpu', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)

    arg_parser.add_argument('--teacher_model_checkpoint', default='', type=str)
    arg_parser.add_argument('--sample_num', default=500000, type=int)
    arg_parser.add_argument('--max_sample_length', default=20, type=int)
    arg_parser.add_argument('--batch_size', default=32, type=int)

    arg_parser.add_argument('--hidden_states', default=256, type=int)
    arg_parser.add_argument('--vocab_size', default=50257, type=int)
    arg_parser.add_argument('--kmeans_iterations', default=1000, type=int)
    arg_parser.add_argument('--pseudocount', default=0.001, type=float)

    arg_parser.add_argument('--output_file', default='hmm.weight', type=str)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


def sample_examples(teacher_model_checkpoint, sample_num, max_sample_length, batch_size):
    teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_checkpoint).to(device)
    eos_token_id = teacher_model.config.eos_token_id
    teacher_model.config.pad_token_id = eos_token_id

    inf = 1e10

    suffixes = [] # sequence_offset, token_offset, token
    suffix_embeddings = []

    for batch_idx in tqdm(range(0, sample_num, batch_size)):
        num_return_seq = min(batch_size, sample_num - batch_idx)
        with torch.no_grad():
            outputs = teacher_model.generate(do_sample=True, min_length=3, max_length=max_sample_length+1,
                num_return_sequences=num_return_seq, top_k=50257,
                output_hidden_states=True, return_dict_in_generate=True)

        sequences = outputs.sequences[:, 1:].clone().to('cpu') # remove the first eos token

        _, d = sequences.shape
        mask = torch.ones(num_return_seq, d).type(torch.LongTensor)
        for i in range(1, d):
            mask[sequences[:, i] == eos_token_id, i] = 0

        token_hidden_states = torch.stack([x[12].clone().to('cpu') for x in outputs.hidden_states], dim=1).squeeze()
        suffix_hidden_states = token_hidden_states * mask.unsqueeze(-1)

        for i in range(0, d):
            suffixes.extend([((batch_idx+j, i), # suffix_offset
                                sequences[j, i].item()) # token
                for j in range(0, num_return_seq) if mask[j, i] == 1])   # suffix_offset = (batch_idx+j, i)
            suffix_embeddings.append(suffix_hidden_states[mask[:, i] == 1, i, :])

    suffix_embeddings = torch.cat(suffix_embeddings, dim=0).detach().cpu().numpy()

    return suffixes, suffix_embeddings


def Kmeans_faiss(vecs, K, max_iterations=1000, nredo=1, verbose=True):
    vecs = numpy.unique(vecs, axis=0) # this line is slow
    kmeans = faiss.Kmeans(vecs.shape[1], K,
        niter=max_iterations, nredo=nredo, verbose=verbose,
        max_points_per_centroid=vecs.shape[0] // K, gpu=True)
    kmeans.train(vecs)

    return kmeans


def update_flows(alpha, beta, gamma, suffixes, idx2cluster,
        hidden_states, vocab_size):

    eos_token_id = vocab_size - 1

    offset2index = {}
    for idx, suffix in enumerate(suffixes):
        offset2index[suffix[0]] = idx

    for idx in tqdm(range(0, len(suffixes))):
        suffix = suffixes[idx]
        suffix_offset, token = suffix
        suffix_offset_next = (suffix_offset[0], suffix_offset[1]+1)
        u = idx2cluster[idx]

        v = None
        if suffix_offset_next in offset2index:
            v = idx2cluster[offset2index[suffix_offset_next]]
        else:
            v = hidden_states - 1 # the reserved hidden state for <eos> token

        alpha[u, v] += 1.0
        beta[u, token] += 1.0
        if suffix_offset[1] == 0:
            gamma[u] += 1.0

    alpha[hidden_states-1, hidden_states-1] = 1.0
    beta[hidden_states-1, eos_token_id] = 1.0    


def write_params(alpha, beta, gamma, pseudocount,
    hidden_states, vocab_size, output_file):

    alpha += pseudocount
    beta += pseudocount
    gamma += pseudocount

    alpha = torch.log(alpha / torch.sum(alpha, dim=-1).unsqueeze(-1))
    beta = torch.log(beta / torch.sum(beta, dim=-1).unsqueeze(-1))
    gamma = torch.log(gamma / torch.sum(gamma, dim=-1))
    
    torch.save({'hidden_states': hidden_states,
            'vocab_size': vocab_size,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,},
        f'{output_file}.th')


def main():
    args = init()

    print(f'sampling {args.sample_num} examples from {args.teacher_model_checkpoint} ...')
    suffixes, suffix_embeddings = sample_examples(args.teacher_model_checkpoint,
        args.sample_num, args.max_sample_length, args.batch_size)

    print(f'training K-means with {args.hidden_states-1} clusters and {len(suffixes)} suffix embeddings ...')
    kmeans = Kmeans_faiss(suffix_embeddings, args.hidden_states - 1,
        max_iterations=args.kmeans_iterations)

    print(f'clustering {len(suffixes)} suffix embeddings into {args.hidden_states-1} clusters ...')
    _, idx2cluster = kmeans.index.search(suffix_embeddings, 1)
    idx2cluster = numpy.squeeze(idx2cluster).tolist()

    alpha = torch.zeros(args.hidden_states, args.hidden_states)
    beta = torch.zeros(args.hidden_states, args.vocab_size)
    gamma = torch.zeros(args.hidden_states)

    print('computing flows ...')
    update_flows(alpha, beta, gamma, suffixes, idx2cluster,
        args.hidden_states, args.vocab_size)

    print(f'storing parameters to {args.output_file} ...')
    write_params(alpha, beta, gamma, args.pseudocount,
        args.hidden_states, args.vocab_size, args.output_file)


if __name__ == '__main__':
    main()