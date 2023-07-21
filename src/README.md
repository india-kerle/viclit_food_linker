## `src/`

This directory contains scripts needed to:

1. **Convert data**. Downsize and convert victorian literature sentences `.csv` into a prodigy-compliant `.jsonl` file. Downsize the food.com dataset to add to knowledge base.
2. **Create a knowledge base.** Create a knowledge base (kb), add aliases. 
3. **Custom data labelling recipe.** Create a custom recipe to manually edit NER labels and generate kb candidates based on extracted FOOD entities. 

### 🛞 Convert data

To downsize and convert data:

```
python src/convert_data.py
```

### 📟 Create a knowledge base

To create a knowledge base, populate it with aliases and prior probabilities and save it locally:

```
python src/create_kb.py
```


## 🌒 Running prodigy recipe 

Run custom recipe locally:

```
prodigy food_linker.manual food_linker_annotated data/training_data.jsonl data/cake_kb -F src/cake_recipe.py 
```

Save labels locally:

```
prodigy db-out food_linker_annotated >> data/food_linker_annotated.jsonl
```