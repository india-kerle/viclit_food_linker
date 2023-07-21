"""
Clean up data to:
    - convert cake recipes data to .jsonl format for prodigy
    - size down cake recipes data

python src/convert_data.py
"""
from pathlib import Path
import pandas as pd
import json
import os

random_state = 43
data_path = Path.cwd() / 'data'

unformatted_training_data = data_path / 'victorian_lit.csv'
cake_recipes = data_path / 'cake_recipes.csv'

if __name__ == '__main__':
    print(f"formatting {unformatted_training_data} to be in .jsonl...")
    training_data_dict = (pd.read_csv(unformatted_training_data)
                          # query food related terms
                          .query("sentences.str.lower().str.contains('cake|cakes')")
                          .reset_index()
                          .rename(columns={'sentences': 'text'})
                          .drop(columns=["label", "index"])
                          .to_dict(orient='records'))

    # dump to jsonl
    with open(data_path / 'training_data.jsonl', 'w') as td:
        for line in training_data_dict:
            json.dump(line, td)
            td.write('\n')

    print(f"size down {cake_recipes}...")
    cake_recipes_df = (pd.read_csv(cake_recipes)
                       [['id', 'name', 'RecipeInstructions']]
                       .sample(10, random_state=random_state))
    cake_recipes_df.to_csv(data_path / 'cake_recipes_sample.csv', index=False)
