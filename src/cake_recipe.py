"""
Custom Prodigy recipe to:
    - edit NER food entities
    - label entities with entity linking based on NER extracted candidate generation 
"""
from pathlib import Path
import copy 
import json
from typing import Dict, Iterator

import spacy
from spacy.kb import InMemoryLookupKB

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens

training_data_path = Path.cwd() / 'data/training_data.jsonl'
kb_path = Path.cwd() / "cake_kb"
entity_loc = Path.cwd() / "data/id_to_name.json"

with entity_loc.open('r') as f:
       id2name_dict = json.load(f)

def make_tasks(nlp: spacy.language.Language,
              kb: spacy.kb.kb.KnowledgeBase,
              stream: Iterator[dict],
              id2name_dict: Dict[str, str] = id2name_dict
              ) -> Iterator[dict]:
    """Add predicted entities and candidate entities to the stream.

    Args:
        nlp (spacy.language.Language): spaCy language model with 
            hf_token_pipe component added to the pipeline
        kb (spacy.kb.kb.KnowledgeBase): spaCy knowledge base
        stream (Iterator[dict]): Iterator of dictionaries with text and label keys
        id2name_dict (Dict[str, str], optional): id to name dictionary. Defaults to id2name_dict.

    Yields:
        Iterator[dict]: Iterator of dictionaries with text, spans, and options keys
    """
    texts = ((eg["text"], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts, as_tuples=True):
        task = copy.deepcopy(eg)
        spans = doc.spans['food-bert']
        ent_labels = []
        for span in spans: 
            # Create a span dict for the predicted entity.
            ent_labels.append(
                {
                    "token_start": span.start,
                    "token_end": span.end - 1,
                    "start": span.start_char,
                    "end": span.end_char,
                    "text": span.text,
                    "label": span.label_,
                }
            )
            candidates = kb.get_candidates(span)
            if candidates:
                options = [{"id": c.entity_, 'text': id2name_dict.get(c.entity_, c.entity_)} for c in candidates]
                options.append({"id": "NIL_otherLink", "text": "Not in options"})
                options.append({"id": "NIL_ambiguous", "text": "Need more context"})
                task["options"] = options
        
        task["spans"] = ent_labels
        
        yield task
            
@prodigy.recipe(
    "food_linker.manual",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .jsonl file", "positional", None, Path),
    kb_loc=("Path to the KB", "positional", None, Path),
)

def entity_linker_manual(dataset, source, kb_loc):
    
    #load blank spacy model
    nlp = spacy.blank("en")
    
    #add food-bert to the pipeline 
    nlp.add_pipe(
        "hf_token_pipe",
        config={
            "model": "Dizex/FoodBaseBERT",
            "annotate": "spans",
            "annotate_spans_key": "food-bert"
        },
    )
    #load the knowledge base from file 
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    
    # Initialize the Prodigy stream 
    stream = JSONL(source)
    
    #add tokens to the stream 
    stream = add_tokens(nlp, stream)
    
    #add predicted entities and candidate entities to the stream 
    stream = make_tasks(nlp, kb, stream)
            
    return {
	    "dataset": dataset,   # save annotations in this dataset
	    "view_id": "blocks", # use the blocks interface
	    "stream": stream,
	     "config": {
	     	"buttons": ["accept", "reject", "ignore"],
	         "labels": ["FOOD"],  # the label for the manual NER interface
	         "blocks": [
                        {"view_id": "ner_manual"},
                        {"view_id": "choice", "text": None},
                        ],
	         "custom_theme": {
	             "labels": {"FOOD": "#2592da"},
	         }
         }
    }