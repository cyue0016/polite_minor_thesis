"""
code requires that GoogleNews-vectors-negative300.bin is in the same directory
This code goes over the first pass, performing two things:
    1. pre-process all requests
    2. turn scores to binary label according to Danescu-Niculescu-Mizil et al. 2013
"""

import tensorflow as tf
import numpy as np
import csv
import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import argparse
import warnings

# Suppress future warnings related to NumPy deprecations
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Stanford Politeness Corpus")
    parser.add_argument(
        "--wiki_file", type=str, default="wikipedia.annotated.csv",
        help="path to WIKI politeness data file")
    parser.add_argument(
        "--se_file", type=str, default="stack-exchange.annotated.csv",
        help="path to SE politeness data file")
    args = parser.parse_args()
    return args

def process_file(file_path):
    request_lst = []
    score_lst = []

    # Read the CSV file using pandas
    df = pd.read_csv(file_path)

    # Extract the required columns
    df = df[['Request', 'Normalized Score']]

    # Convert DataFrame columns to numpy arrays
    requests = df['Request'].values
    scores = df['Normalized Score'].values

    # Iterate over numpy arrays directly
    for request, score in zip(requests, scores):
        try:
            request_lst.append(request)
            score_lst.append(score)
        except Exception as e:
            print(f"Error processing request and score: {e}")

    # Calculate the 25th and 75th percentiles
    score_25 = np.percentile(score_lst, 25)
    score_75 = np.percentile(score_lst, 75)

    # Convert scores to binary labels and filter requests
    combined = [
        (request, (score > score_75) * 1, score)
        for (request, score) in zip(request_lst, score_lst)
        if (score < score_25 or score > score_75)
    ]

    if combined:
        requests, labels, scores = zip(*combined)
        data_lst = [list(requests), list(labels)]
    else:
        data_lst = [[], []]

    return data_lst


# Main processing logic
base_dir = "C:/Users/Kelvin/Documents/Research/polite-dialogue-generation-master/data/Stanford_politeness_corpus"
current_dir = "data/Stanford_politeness_corpus/"
args = parse_args()
file_names = [args.wiki_file, args.se_file]

for file_name in file_names:
    file_path = os.path.join(current_dir, file_name)
    data_lst = process_file(file_path)

    processed_file_names = [
        prefix + file_name[:(-4)] + ".pkl"
        for prefix in ["dataset_", "labels_"]
    ]

    processed_file_paths = [
        os.path.join(current_dir, processed_file_name)
        for processed_file_name in processed_file_names
    ]

    for (processed_file_path, data) in zip(processed_file_paths, data_lst):
        with open(processed_file_path, 'wb') as fp:
            pickle.dump(data, fp)
            print(f"Done pickling {processed_file_path}")
