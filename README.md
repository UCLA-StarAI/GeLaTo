# GeLaTo


```
conda create --name gelato python=3.8
conda activate gelato
pip3 install torch torchvision torchaudio transformers==4.21.3 datasets lemminflect
conda install -c pytorch faiss-gpu
```

# for eval
```
pip3 install evaluate rouge_score
pip3 install -U spacy
python -m spacy download en_core_web_sm
```
