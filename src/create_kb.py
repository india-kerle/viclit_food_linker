"""
Script to create a dummy knowledge base from a CSV file of entities.

python src/create_kb.py
"""

import csv
from pathlib import Path

from spacy.kb import InMemoryLookupKB

import spacy
from typing import Dict, Union
import json


data_path = Path.cwd() / 'data'
entity_path = data_path / 'cake_recipes_sample.csv'

nlp = spacy.load('en_core_web_sm')

def load_entities(entity_path: Path = entity_path) -> Union[Dict[str, str], Dict[str, str]]:
    """Load entities from a CSV file.

    Args:
        entity_path (Path): Path to the CSV file.

    Returns:
        Union[Dict[str, str], Dict[str, str]]: A dictionary of entity QIDs to entity names 
            and a dictionary of entity QIDs to entity descriptions.
    """
        
    names = dict()
    descriptions = dict()
    
    with entity_path.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
            
    return names, descriptions

def create_dummy_kb(nlp : spacy.language.Language, 
                    desc_dict: Dict[str, str], 
                    name_dict: Dict[str, str] ) -> InMemoryLookupKB:
    """Create knowledge base from a dictionary of entity QIDs to entity descriptions 
        and a dictionary of entity QIDs to entity names.

    Args:
        nlp (spacy.language.Language): spacy language model
        desc_dict (Dict[str, str]): Dictionary of entity QIDs to entity descriptions
        name_dict (Dict[str, str]): Dictionary of entity QIDs to entity names

    Returns:
        InMemoryLookupKB: A knowledge base with entities and aliases added.
    """

    #instantiate a knowledge base and add entities
    kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=96)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)  # 342 is an arbitrary value here

    qids = name_dict.keys()
    probs = [1/(len(qids) + 1) for qid in qids] # to make sure its less than 1 
    
    cake_terms = ['cake', 'cakes', '-cakes', 'glazed tarts', 'petrified sponge-cake', 'toasted oat-cake', 'nut cakes', '-cake']
    for cake_term in cake_terms:
        kb.add_alias(alias=cake_term, entities=qids, probabilities=probs)  # sum([probs]) should be <= 1 !
    
    return kb

if __name__ == "__main__":
    
    print('loading entities...')
    names, descriptions = load_entities()
    names.pop('id')
    descriptions.pop('id')
    
    print('creating knowledge base...')
    kb = create_dummy_kb(nlp, descriptions, names)
    
    print('saving knowledge base and id to name file ...')
    kb.to_disk(data_path / "cake_kb")
    #save names description to json to /data folder
    with open(data_path / 'id_to_name.json', 'w') as f:
        json.dump(names, f) 