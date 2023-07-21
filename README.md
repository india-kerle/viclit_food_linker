## 🎂 Disambiguating VicLit cake mentions 

This repo contains the scripts needed to generate training data to train an entity linker to disambiguate mentions of cake in victorian literature
to food.com cake recipes (and generate more training data to improve [FoodBERT](https://huggingface.co/Dizex/FoodBaseBERT-NER), a fine-tuned BERT model to use for Named Entity Recognition of Food entities).

This is purely illustrative to experiment with: 

1. Writing a custom prodigy recipe;
2. `spacy-huggingface-pipelines`;
3. knowledge bases

## 🔨 Set Up

**Download spacy model:**

```
python -m spacy download en_core_web_sm
```

**Install dependencies:**
```
conda create --name india_prodigy pip python=3.9 #create conda environment
conda activate india_prodigy #activate conda environment 
conda install pytorch torchvision torchaudio -c pytorch-nightly #install pytorch etc. 
pip install prodigy -f https://[YOUR_LICENSE_KEY]@downloadprodi.gy #install prodigy with license key
pip install -r requirements.txt #install other dependencies 
```

## 🪜 Possible next steps 

1. Train entity linker
2. further fine-tune FoodBERT

