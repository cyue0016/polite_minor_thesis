#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import pickle


def load_pickle(filename):
    with open(filename, "rb") as fp:
        lst = pickle.load(fp)
    print(f"Done loading {filename}.")
    return lst


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply Politeness Model to Forum Data")
    parser.add_argument("--ckpt", type=str, default="ckpt/politeness_classifier_2",
                        help="Path to the trained checkpoint")
    parser.add_argument("--input_file", type=str,
                        default="data/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.xlsx",
                        help="Path to the input Excel file")
    parser.add_argument("--output_file", type=str, default="data/stanfordMOOCForumPostsSet/with_scores.csv",
                        help="Path to save the CSV with politeness scores")
    parser.add_argument("--tokenized_forum_file", type=str,
                        default="data/stanfordMOOCForumPostsSet/tokenized_forum.pkl",
                        help="Path to tokenized forum posts .pkl file")
    args = parser.parse_args()
    return args


def load_data(args):
    # Load tokenized forum posts from the .pkl file
    print(f"Loading tokenized forum data from {args.tokenized_forum_file}")
    forum_data = load_pickle(args.tokenized_forum_file)
    print(f"Done loading {args.tokenized_forum_file}.")

    # Load original Excel file to later add politeness scores
    print(f"Loading Excel file {args.input_file}")
    df = pd.read_excel(args.input_file, engine='openpyxl')
    return forum_data, df


def pad(input_seqs, sequence_lengths):
    """ Pad input sequences to the maximum length """
    max_length = max(sequence_lengths)
    padded = [seq + [0] * (max_length - len(seq)) for seq in input_seqs]
    return padded


def find_tensor_by_partial_name(graph, partial_name):
    """ Helper function to find a tensor in the graph by a partial name match """
    for tensor in graph.as_graph_def().node:
        if partial_name in tensor.name:
            return tensor.name + ":0"
    raise ValueError(f"Tensor with partial name {partial_name} not found in the graph")


def main(args):
    forum_data, df = load_data(args)

    # Load the trained model
    print("Building model...")
    ckpt_path = args.ckpt
    graph = tf.Graph()

    with graph.as_default():
        # Force model to use CPU if GPU is unavailable
        with tf.device('/cpu:0'):  # Ensure everything runs on CPU
            saver = tf.train.import_meta_graph(ckpt_path + ".meta")

        with tf.Session(graph=graph) as sess:
            with tf.device('/cpu:0'):  # Ensure session uses CPU
                saver.restore(sess, ckpt_path)
                print("Restored model from checkpoint.")

                # Dynamically find tensors by partial name match
                inputs = graph.get_tensor_by_name(find_tensor_by_partial_name(graph, "inputs"))
                seq_lengths = graph.get_tensor_by_name(find_tensor_by_partial_name(graph, "seq_lengths"))
                is_training = graph.get_tensor_by_name(find_tensor_by_partial_name(graph, "is_training"))
                batch_scores = graph.get_tensor_by_name(find_tensor_by_partial_name(graph, "batch_scores"))

                # Prepare data for scoring
                num_sents = len(forum_data)
                batch_size = 32
                num_batches = num_sents // batch_size

                all_scores = []
                for i in range(num_batches):
                    start = i * batch_size
                    end = start + batch_size
                    input_seqs = forum_data[start:end]
                    sequence_lengths = [len(seq) for seq in input_seqs]

                    feed_dict = {
                        inputs: pad(input_seqs, sequence_lengths),
                        seq_lengths: sequence_lengths,
                        is_training: False
                    }

                    scores = sess.run(batch_scores, feed_dict=feed_dict)
                    all_scores.extend(scores)

                # Append any leftover sentences (if num_sents % batch_size != 0)
                if num_sents % batch_size != 0:
                    input_seqs = forum_data[num_batches * batch_size:]
                    sequence_lengths = [len(seq) for seq in input_seqs]
                    feed_dict = {
                        inputs: pad(input_seqs, sequence_lengths),
                        seq_lengths: sequence_lengths,
                        is_training: False
                    }
                    scores = sess.run(batch_scores, feed_dict=feed_dict)
                    all_scores.extend(scores)

                # Append politeness scores to DataFrame
                print("Assigning politeness scores to the Excel data.")
                df['politeness_score'] = all_scores

                # Save the updated dataframe as a CSV
                print(f"Saving results to {args.output_file}")
                df.to_csv(args.output_file, index=False)
                print(f"Saved politeness scores to {args.output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
