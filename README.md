# GeLaTo

This is the source code for the paper ["Tractable Control for Autoregressive Language Generation"](https://arxiv.org/abs/2304.07438) (ICML 2023)


## Requirements
We suggest using conda to setup environment. 

```
conda create --name gelato python=3.8
conda activate gelato
```

for PyTorch & Transformers:
```
pip3 install torch torchvision torchaudio transformers==4.21.3 datasets lemminflect
conda install -c pytorch faiss-gpu
```

to train HMMs with Juice.jl, you need to download Julia:
```
https://julialang.org/downloads/
```

for evaluation:
```
pip3 install evaluate rouge_score
pip3 install -U spacy
python -m spacy download en_core_web_sm
```

## Models & Outputs
We release checkpoints for the base models (GPT2-large finetuned on CommonGen) and the distilled HMMs for reproducibility. In addition, we also release the generated examples.

```
https://drive.google.com/drive/folders/1cagRWGrGQ6HNes0z7Li2dHo2PfcuuZEl?usp=sharing
```

## Running the GeLaTo Pipeline

We use CommonGen (unsupervised setting) as an example to illustrate how to run the GeLaTo pipeline. See contents of the scripts for full command lines.

### 1. finetuning the base model
```
bash scripts/1_finetune_gpt.sh
```


### 2. training the HMMs
To train an HMM that approximates the base model, there are three steps:

* sampling training data from the base model 
    ```
    bash scripts/2_sample_training_data.sh
    ```

* using latent variable distillation (LVD) to initialize HMM parameters
    ```
    bash scripts/3_lvd_hmm.sh
    ```

* train HMM with EM (need Julia installation)
    ```
    bash scripts/4_train_hmm.sh
    ```

### 3. generation
```
bash scripts/5_decode.sh
```

### 4. re-ranking the generated sentences
```
bash scripts/6_select_sentence.sh
```

### 5. evaluation
```
bash scripts/download_eval_dependencies.sh
bash scripts/7_evaluate.sh
```


