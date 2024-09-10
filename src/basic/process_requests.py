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
        "--se_file", type=str, default="data/Stanford_politeness_corpus/labels_stack-exchange.annotated.pkl",
        help="path to SE politeness data file")
    parser.add_argument(
        "--forum_file", type=str, default="data/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.xlsx",
        help="Path to Forum Posts Excel file")
    parser.add_argument(
        "--forum_text_column", type=str, default="Text",
        help="The column name that contains the forum text data")
    parser.add_argument(
        "--tagger_path", type=str, 
        default="stanford-postagger-full-2020-11-17/stanford-postagger.jar",
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
Handle forum post Excel file if provided
"""
if forum_file:
    df = pd.read_excel(forum_file, engine='openpyxl')

    if forum_text_column not in df.columns:
        raise ValueError(f"Text column '{forum_text_column}' not found in the forum file.")

    texts = df[forum_text_column].tolist()
    tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
    # Ensure that only valid strings are tokenized
    tokenized_forum_posts = [tweet_tokenizer.tokenize(str(text)) for text in texts if isinstance(text, str)]

    # Convert all tokens to lowercase
    tokenized_forum_posts = [[token.lower() for token in post] for post in tokenized_forum_posts]

    # Append forum posts to datasets for further processing
    datasets.append(tokenized_forum_posts)

# Ensure that text is decoded from bytes to string if necessary
def decode_if_bytes(text):
    if isinstance(text, bytes):
        return text.decode('utf-8')  # Adjust the encoding as per your data
    return text

print("Tokenizing all requests.")

tweet_tokenizer = TweetTokenizer(
    preserve_case=True, reduce_len=True, strip_handles=True)

# Apply decoding and tokenization
tokenized_datasets_original_tweet = [
    [tweet_tokenizer.tokenize(decode_if_bytes(request)) for request in dataset]
    for dataset in datasets
]

print("Retokenizing with Stanford tokenizer. This may take a long time.")

path_pos = "/stanford-postagger-full-2020-11-17/"
jar_pos = "stanford-postagger.jar"

tokenizer = StanfordTokenizer(path_pos + jar_pos)

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

UNK = "UNK_TOKEN"

if use_existing_vocab:
    vocab_politeness = load_pickle(os.path.join(output_dir, "vocab_politeness.pkl"))
else:
    # Load word embedding model
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = KeyedVectors.load_word2vec_format(fname=word2vec, binary=True)

    freq_threshold = 2

    # Get all tokens from all datasets
    all_tokens = [token for dataset in datasets for request in dataset for token in request]

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


print("Start indexing datasets... This may take a while")
indexed_datasets = [
    [[replace_with_index(token) for token in request] for request in dataset]
    for dataset in datasets
]

"""
Save the processed data
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

    if forum_file:
        lsts.append(indexed_datasets[-1])  # Adding indexed forum dataset
        pickle_lst.append("dataset_forum.pkl")  # Save forum as 'dataset_forum.pkl'

# Hardcoding file paths for saving
pickle_files = [
    "data/Stanford_politeness_corpus/vocab_politeness.pkl",
    "data/Stanford_politeness_corpus/shared_vocab_politeness.pkl",
    "data/Stanford_politeness_corpus/new_vocab_politeness.pkl",
    "data/Stanford_politeness_corpus/dataset_WIKI.pkl",
    "data/Stanford_politeness_corpus/dataset_SE.pkl",
    "data/Stanford_politeness_corpus/embedding_word2vec_politeness.pkl"
]

if forum_file:
    pickle_files.append("data/Stanford_politeness_corpus/dataset_forum.pkl")  # Add hardcoded forum save path
    pickle_files.append("data/stanfordMOOCForumPostsSet/dataset_forum.pkl")

# Save the processed data to the hardcoded paths
dump_pickles(pickle_files, lsts)

print(f"Preprocessing complete. Files saved to the hardcoded paths.")
