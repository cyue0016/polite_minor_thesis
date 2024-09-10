#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk import FreqDist
import logging
import re
from gensim.models import KeyedVectors
import string
import pickle
import random
from nltk.tokenize.stanford import StanfordTokenizer
import argparse
from util import load_pickle, load_pickles, dump_pickles, dump_pickle


# In[ ]:
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Forum Post Data")
    parser.add_argument(
        "--xlsx_file", type=str, required=True,
        help="path to the Excel file containing the forum post data")
    parser.add_argument(
        "--text_column", type=str, required=True,
        help="the column name that contains the text data to be tokenized")
    parser.add_argument(
        "--use_existing_vocab", action="store_true",
        help="whether to use an existing vocab set")
    args = parser.parse_args()
    return args

args = parse_args()
xlsx_file = args.xlsx_file
text_column = args.text_column
use_existing_vocab = args.use_existing_vocab

# Load the data from the xlsx file
df = pd.read_excel(xlsx_file, engine='openpyxl')

# Inspect the first few rows of the dataset
print(f"Loaded data from {xlsx_file}. First few rows:")
print(df.head())

# Ensure the text_column exists in the dataframe
if text_column not in df.columns:
    raise ValueError(f"The specified text column '{text_column}' is not found in the dataset.")

# Select the text column for tokenization
texts = df[text_column].tolist()

# Tokenize the text using TweetTokenizer
tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
tokenized_texts = [tweet_tokenizer.tokenize(text) for text in texts]

# Example: Convert the text to lowercase
tokenized_texts = [[token.lower() for token in text] for text in tokenized_texts]

# Build vocabulary
UNK = "UNK_TOKEN"
if use_existing_vocab:
    # Load existing vocab if needed
    vocab_politeness = load_pickle("vocab_politeness.pkl")
else:
    # Create a new vocabulary from the tokenized texts
    all_tokens = [token for text in tokenized_texts for token in text]
    vocab_politeness = list(set(all_tokens))
    vocab_politeness = [UNK] + vocab_politeness

# Create dictionaries for token-to-index and index-to-token mappings
token2index = {token: i for i, token in enumerate(vocab_politeness)}
index2token = {i: token for i, token in enumerate(vocab_politeness)}

# Replace tokens in texts with their corresponding indices
def replace_with_index(token):
    return token2index.get(token, token2index[UNK])

indexed_texts = [[replace_with_index(token) for token in text] for text in tokenized_texts]

# Now let's save the processed data
data_path = "data/processed_forum_posts"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Save the vocab and tokenized/indexed data
pickle_files = [
    os.path.join(data_path, "vocab_politeness.pkl"),
    os.path.join(data_path, "indexed_texts.pkl"),
    os.path.join(data_path, "forum_data.pkl")  # Original forum data
]

# Dump vocab, indexed texts, and the original dataframe as pickle files
dump_pickles(pickle_files, [vocab_politeness, indexed_texts, df])

print(f"Preprocessing complete. Files saved to: {data_path}")
