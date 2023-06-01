from eval_metrics.bleu import Bleu
from eval_metrics.cider import Cider
from eval_metrics.spice import Spice

# import evaluate
import spacy
import json
# import sys
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_file', default="", type=str)
parser.add_argument('--result_file', default="", type=str)
args = parser.parse_args()

nlp = spacy.load("en_core_web_sm")

def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp(sentence):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list

    return dict


def evaluator(gts, res):
    eval = {}
    # =================================================
    # Set up scorers
    # =================================================
    print('tokenization...')
    # Todo: use Spacy for tokenization
    gts = tokenize(gts)
    res = tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        print("computing %s score..." % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                eval[m] = sc
                print("%s: %0.3f" % (m, sc))
        else:
            eval[method] = score
            print("%s: %0.3f" % (method, score))

# 
def load_targets(dataset_file):
    with open(dataset_file, 'r') as fin:
        examples = json.load(fin)
        
    examples_ = {}
    for example in examples:
        idx = example['concept_set_idx']
        if idx in examples_:
            examples_[idx]['sentences'] = examples_[idx]['sentences'] + [example['target']]
        else:
            examples_[idx] = {
                'concept_set_idx': idx,
                'concepts': example['concepts'],
                'sentences': [example['target']],
            }
    
    examples = [v for _, v in examples_.items()]
    
    return examples

targets = load_targets(args.target_file)

with open(args.result_file, 'r') as fin:
    results = json.load(fin)

targets = sorted(targets, key=lambda x:x['concept_set_idx'])
results = sorted(results, key=lambda x:x['concept_set_idx'])

results_idx_set = set([example['concept_set_idx'] for example in results])
targets = [example for example in targets if example['concept_set_idx'] in results_idx_set]

gts = {}
res = {}
for gts_line, res_line in zip(targets, results):
    assert(gts_line['concepts'] == res_line['concepts'])
    key = '#'.join(gts_line['concepts'])
    gts[key] = [x.rstrip('\n') for x in gts_line['sentences']]

    sentence = res_line['sentence']
    sentence.replace('.', ' .')
    sentence.replace(',', ' ,')
    res[key] = [sentence.rstrip('\n')]
    # res[key] = [res_line['sentence'].rstrip('\n')]

evaluator(gts, res)

# print("Evaluation from huggingface evaluate")


from rouge_score import rouge_scorer
predictions = [x['sentence'] for x in result]
references = [x['sentences'] for x in targets]
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

scores = []
for pred, ref in zip(predictions, references):
    rs = [scorer.score(pred, i)['rougeL'].fmeasure for i in ref]
    scores.append(sum(rs)/len(rs))
    # scores.append(max(rs))
print('rougeL score')
print(sum(scores) / len(scores))


