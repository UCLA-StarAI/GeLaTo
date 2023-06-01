import os
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = 'cuda'


class DatasetFromFile(torch.utils.data.Dataset):
  def __init__(self, dataset_file, seq2seq):
    with open(dataset_file, 'r') as fin:
        data = json.load(fin)

    if seq2seq:
        texts = ['<|endoftext|>' + ' ' + ' '.join(e['concepts']) + ' = ' + e['target'] + '<|endoftext|>' for e in data]
    else:
        texts = ['<|endoftext|>' + ' ' + e['target'] + '<|endoftext|>' for e in data]

    self.texts = texts

  def __len__(self):
        return len(self.texts)

  def __getitem__(self, index):
        return self.texts[index]


def init():
    global device

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)

    arg_parser.add_argument('--max_epoch', default=20, type=int)
    arg_parser.add_argument('--batch_size', default=8, type=int)
    arg_parser.add_argument('--lr', default=0.0001, type=float)
    arg_parser.add_argument('--grad_accum_iters', default=1, type=int)
    arg_parser.add_argument('--max_sequence_length', default=None, type=int)

    arg_parser.add_argument('--seq2seq', action='store_true')
    arg_parser.add_argument('--skip_eval', action='store_true')

    arg_parser.add_argument('--train_data_file', default='common_gen', type=str)
    arg_parser.add_argument('--validation_data_file', default='', type=str)
    arg_parser.add_argument('--model_path', default='', type=str)
    arg_parser.add_argument('--log_file', default='log.txt', type=str)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


def aggregate_loss(model, data_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            loss = model(**inputs).loss
            losses.append(loss.item())
    return torch.mean(torch.Tensor(losses)).item()


def main():
    args = init()

    train_data = DatasetFromFile(args.train_data_file, args.seq2seq)
    if args.validation_data_file != '':
        validation_data = DatasetFromFile(args.validation_data_file, args.seq2seq)
    else:
        validation_data = None

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    tokenizer.pad_token = tokenizer.eos_token

    def collate(batch):
        batch_encoding = tokenizer([text for text in batch], padding=True)

        labels = [[(x if y == 1 else -100) for x, y in zip(e, mask)]
            for e, mask in zip(batch_encoding['input_ids'], batch_encoding['attention_mask'])]
        batch_encoding_tensor = {
            'input_ids': torch.LongTensor(batch_encoding['input_ids']),
            'attention_mask': torch.LongTensor(batch_encoding['attention_mask']),
            'labels': torch.LongTensor(labels)
        }

        if (args.max_sequence_length is not None) and \
            (args.max_sequence_length < batch_encoding_tensor['input_ids'].shape[1]):
            batch_encoding_tensor = {
                'input_ids': batch_encoding_tensor['input_ids'][:, :args.max_sequence_length],
                'attention_mask': batch_encoding_tensor['attention_mask'][:, :args.max_sequence_length],
                'labels': batch_encoding_tensor['labels'][:, :args.max_sequence_length]
            }

        return batch_encoding_tensor

    train_loader = DataLoader(train_data, collate_fn=collate, batch_size=args.batch_size, shuffle=True)
    if validation_data is not None:
        validation_loader = DataLoader(validation_data, collate_fn=collate, batch_size=args.batch_size, shuffle=True)

    model = GPT2LMHeadModel.from_pretrained('gpt2-large',
        pad_token_id=tokenizer.eos_token_id)
    print('Saving checkpoint-0')
    model_save_path = os.path.join(args.model_path, 'checkpoint-0')
    model.save_pretrained(model_save_path)

    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.max_epoch+1):
        print(f'epoch {epoch}')

        model.train()
        optim.zero_grad()
        batch_idx = 0
        for batch in tqdm(train_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            loss = model(**inputs).loss

            loss = loss / args.grad_accum_iters

            loss.backward()

            batch_idx += 1
            if (batch_idx % args.grad_accum_iters == 0) or (batch_idx == len(train_loader)):
                optim.step()
                optim.zero_grad()


        print(f'Saving checkpoint-{epoch}')
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        model_save_path = os.path.join(args.model_path, f'checkpoint-{epoch}')
        model.save_pretrained(model_save_path)

        if not args.skip_eval:
            print(f'Evaluating checkpoint-{epoch}')
            train_loss = aggregate_loss(model, train_loader)
            if validation_data is not None:
                validation_loss = aggregate_loss(model, validation_loader)
            else:
                validation_loss = -1.0

            msg = f'epoch {epoch}, train_loss: {train_loss}, validation_loss: {validation_loss}'

            print(msg)
            with open(args.log_file, 'a+') as fout:
                fout.write(msg + '\n')
        else:
            print('Skipped evaluation')


if __name__ == '__main__':
    main()
