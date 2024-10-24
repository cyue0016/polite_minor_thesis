#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
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
        description="Preprocess Stanford Politeness Corpus and Forum Post Data")
    parser.add_argument(
        "--wiki_file", type=str, default="data/Stanford_politeness_corpus/dataset_wikipedia.annotated.pkl",
        help="path to WIKI politeness data file")
    parser.add_argument(
        "--se_file", type=str, default="data/Stanford_politeness_corpus/dataset_stack-exchange.annotated.pkl",
        help="path to SE politeness data file")
    parser.add_argument(
        "--forum_file", type=str, default="data/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.xlsx",
        help="Path to Forum Posts Excel file")
    parser.add_argument(
        "--forum_text_column", type=str, default="Text",
        help="The column name that contains the forum text data")
    parser.add_argument(
        "--tagger_path", type=str, 
        default="stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar",
        help="path to the Stanford pos tagger")
    parser.add_argument(
        "--word2vec", type=str, default="GoogleNews-vectors-negative300.bin",
        help="path to pretrained word2vec binary file")
    parser.add_argument(
        "--use_existing_vocab", action="store_true", 
        help="whether to use an existing vocab set")
    args = parser.parse_args()
    return args


"""
Load file
"""
# Call parse_args to initialize the args variable
args = parse_args()

wiki_file = args.wiki_file
se_file = args.se_file
forum_file = args.forum_file
forum_text_column = args.forum_text_column
tagger_path = args.tagger_path
word2vec = args.word2vec
use_existing_vocab = args.use_existing_vocab


# Load datasets if provided
files = []
if wiki_file:
    files.append(wiki_file)
if se_file:
    files.append(se_file)

datasets = load_pickles(files) if files else []

"""
Handle forum post Excel file separately
"""
if forum_file:
    df = pd.read_excel(forum_file, engine='openpyxl')

    if forum_text_column not in df.columns:
        raise ValueError(f"Text column '{forum_text_column}' not found in the forum file.")

    texts = df[forum_text_column].tolist()
    tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)

    processed_forum_posts = []

    for i, text in enumerate(texts):
        if isinstance(text, list):
            # If it's a list, assume it's already tokenized and lowercase the tokens
            processed_forum_posts.append([token.lower() for token in text])
        elif isinstance(text, str):
            # If it's a string, tokenize and lowercase as usual
            tokenized_post = tweet_tokenizer.tokenize(text)
            processed_forum_posts.append([token.lower() for token in tokenized_post])
        elif isinstance(text, (int, float)):
            # Convert int and float to string before tokenizing
            tokenized_post = tweet_tokenizer.tokenize(str(text))
            processed_forum_posts.append([token.lower() for token in tokenized_post])
        else:
            # If it can not be handled, send the message.
            print(f"Line {i}: Skipping unrecognized type: {type(text)}")

    # Save the forum posts separately without adding to datasets used for model training
    forum_tokenized_path = "data/stanfordMOOCForumPostsSet/tokenized_forum.pkl"
    dump_pickle(forum_tokenized_path, processed_forum_posts)
    print(f"Forum posts tokenized and saved to {forum_tokenized_path}")

"""
Continue tokenizing WIKI and SE datasets
"""
print("Tokenizing all requests from Wikipedia and Stack Exchange datasets.")

tweet_tokenizer = TweetTokenizer(
    preserve_case=True, reduce_len=True, strip_handles=True)

# Continue processing the rest of the datasets (Wiki and SE)
tokenized_datasets_original_tweet = [
    [tweet_tokenizer.tokenize(request) if isinstance(request, str) else request
     for request in dataset]
    for dataset in datasets
]

print("Retokenizing with Stanford tokenizer. This may take a long time.")

tokenizer = StanfordTokenizer(tagger_path)

tokenized_datasets_original = [
    [tokenizer.tokenize(' '.join(request).strip())
     for request in dataset]
    for dataset in tokenized_datasets_original_tweet]
# tokenized_datasets_original = tokenized_datasets_original_tweet

"""
Convert all tokens to lowercase
"""
tokenized_datasets = [
    [[token.lower()
      for token in request]
     for request in dataset]
    for dataset in tokenized_datasets_original]

"""
Build the whole vocabulary

Vocab lists:
• special token: "UNK_TOKEN"
• vocab_shared: intersection of word2vec vocab and politeness vocab
• vocab_freq: frequent vocab that is not in word2vec vocab
"""
output_dir = "data/Stanford_politeness_corpus/"

UNK = "UNK_TOKEN"

if use_existing_vocab:
    vocab_politeness = load_pickle(os.path.join(output_dir, "vocab_politeness.pkl"))
else:
    # Load word embedding model
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = KeyedVectors.load_word2vec_format(fname=word2vec, binary=True)

    freq_threshold = 2

    # Get all tokens from the WIKI and SE datasets
    all_tokens = [token for dataset in tokenized_datasets for request in dataset for token in request]

    fdist = FreqDist(all_tokens)
    fdist_lst = fdist.most_common()
    vocab_politeness = [token for (token, _) in fdist_lst]
    vocab_politeness_freq = [
        token for (token, freq) in fdist_lst if freq >= freq_threshold
    ]

    vocab_word2vec = list(model.key_to_index)  # get word2vec vocabulary list
    vocab_shared = list(set(vocab_politeness).intersection(set(vocab_word2vec)))
    vocab_new = list(set(vocab_politeness_freq).difference(set(vocab_word2vec)))
    vocab_politeness = [UNK] + vocab_new + vocab_shared

    print(f"Shared vocab size: {len(vocab_shared)}")
    print(f"New vocab size: {len(vocab_new)}")

    # Obtain the reduced word2vec embedding matrix
    embedding_word2vec = model[vocab_shared]

"""
Create dictionaries between indices and tokens
"""
index2token = {i: token for (i, token) in enumerate(vocab_politeness)}
token2index = {token: i for (i, token) in enumerate(vocab_politeness)}

"""
Replace a token with its index in the vocab
"""
index_UNK = token2index[UNK]


def replace_with_index(token):
    try:
        return token2index[token]
    except:
        return index_UNK


print("Start indexing WIKI and SE datasets... This may take a while")
indexed_datasets = [
    [[replace_with_index(token) for token in request] for request in dataset]
    for dataset in tokenized_datasets
]

"""
Save the processed data (WIKI and SE datasets only)
"""
if use_existing_vocab:
    lsts = [indexed_datasets[0], indexed_datasets[1]]  # WIKI and SE datasets
    pickle_lst = ["dataset_WIKI.pkl", "dataset_SE.pkl"]
else:
    lsts = [
        vocab_politeness,
        vocab_shared,
        vocab_new,
        indexed_datasets[0], indexed_datasets[1],
        embedding_word2vec
    ]
    pickle_lst = [
        "vocab_politeness.pkl",
        "shared_vocab_politeness.pkl",
        "new_vocab_politeness.pkl",
        "dataset_WIKI.pkl",
        "dataset_SE.pkl",
        "embedding_word2vec_politeness.pkl"
    ]

# Save the WIKI and SE processed data
pickle_files = [
    "data/Stanford_politeness_corpus/vocab_politeness.pkl",
    "data/Stanford_politeness_corpus/shared_vocab_politeness.pkl",
    "data/Stanford_politeness_corpus/new_vocab_politeness.pkl",
    "data/Stanford_politeness_corpus/dataset_WIKI.pkl",
    "data/Stanford_politeness_corpus/dataset_SE.pkl",
    "data/Stanford_politeness_corpus/embedding_word2vec_politeness.pkl"
]

# Save the processed data to the hardcoded paths
dump_pickles(pickle_files, lsts)

print(f"Preprocessing complete. Files for WIKI and SE datasets saved to the hardcoded paths.")
