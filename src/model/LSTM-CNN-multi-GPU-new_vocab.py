#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports for compatibility between Python 2&3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import tensorflow as tf
import numpy as np
import pickle
import os
import random
import itertools
from nltk.tokenize import word_tokenize
import argparse
import pandas as pd
import sys
sys.path.append(".")
from src.basic.util import (decode2string, remove_duplicates, 
                            zip_lsts, unzip_lst, 
                            dump_pickle, dump_pickles, 
                            load_pickle, load_pickles,
                            read_lines, write_lines)


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Stanford Politeness Corpus")
    parser.add_argument(
        "--data_path", type=str, default="data/Stanford_politeness_corpus/",
        help="path to WIKI politeness data file")
    parser.add_argument(
        "--test", action="store_true",
        help="whether we are testing")
    parser.add_argument(
        "--ckpt", type=str, default="ckpt/politeness_classifier_2")
    parser.add_argument(
        "--use_all_data", action="store_true",
        help="whether we use all data (including the test data) for training")
    # New arguments for Excel file input and output
    parser.add_argument(
        "--tokenized_file", type=str, default="data/stanfordMOOCForumPostsSet/tokenized_forum.pkl",
        help="Path to the tokenized Excel file containing the text data")
    parser.add_argument(
        "--output_file", type=str, default="data/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.csv",
        help="Path to save the csv file with politeness scores")
    parser.add_argument(
        "--forum_file", type=str, default="data/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.xlsx",
        help="Path to the Excel file containing the text data")

    args = parser.parse_args()
    return args


args = parse_args()
data_path = args.data_path
is_test = args.test
ckpt = args.ckpt
use_all_data = args.use_all_data
model_path = "ckpt/"

batch_size = 32
num_gpus = 1
string_input = False # whether inputs are strings or indecies


"""
Load pickled lists
"""
filenames = [
    "vocab_politeness",  "shared_vocab_politeness", "new_vocab_politeness",
    "dataset_WIKI", "dataset_SE",
    "labels_wikipedia.annotated", "labels_stack-exchange.annotated",
    "embedding_word2vec_politeness",
]

files = [os.path.join(data_path, filename + ".pkl") 
         for filename in filenames]

data = load_pickles(files)

vocab_politeness = data[0]
shared_vocab_politeness = data[1]
new_vocab_politeness = data[2]

vocab_size_politeness = len(vocab_politeness)
shared_vocab_size_politeness = len(shared_vocab_politeness)
new_vocab_size_politeness = len(new_vocab_politeness)

# +1 for "UNK_TOKEN"
assert vocab_size_politeness == 1 + new_vocab_size_politeness + shared_vocab_size_politeness
print("Classifier vocab size: %d" % vocab_size_politeness)

requests_WIKI = data[3]
requests_SE = data[4]
labels_WIKI = data[5]
labels_SE = data[6]

embedding_word2vec_politeness = data[7]


# Index vocabulary
index2token = {i: token for (i, token) in enumerate(vocab_politeness)}
token2index = {token: i for (i, token) in enumerate(vocab_politeness)}


hidden_size = 256
embedding_size = 300
num_classes = 2
# Add the number of epochs to 3
num_epochs = 3
batch_size_per_gpu = batch_size // num_gpus

# hyper-parameters
learning_rate = 0.0005
filter_sizes = [3, 4, 5]
num_filters = 75
dropout_rate = 0.5
dropout_rate_lstm = 0.5
clipping_threshold = 5.0


"""
Shuffle two lists without changing their correspondences
Args:
    lst1: list 1
    lst2: list 2
Returns:
    The two shuffled lists
"""
def shuffle(lst1, lst2):
    combined = list(zip(lst1, lst2))
    np.random.shuffle(combined)
    (shuffled_lst1, shuffled_lst2) = zip(*combined)
    return [list(shuffled_lst1), list(shuffled_lst2)]

def compute_n_examples(n_total, percentile):
    return np.floor(
        np.percentile(np.arange(n_total), percentile)).astype(np.int32)

num_requests_WIKI = len(requests_WIKI)
num_requests_SE = len(requests_SE)
print("num_requests_WIKI", num_requests_WIKI)
print("num_requests_SE", num_requests_SE)

num_test_batches_WIKI = compute_n_examples(num_requests_WIKI, 20) // batch_size
num_test_WIKI = num_test_batches_WIKI * batch_size
num_test_batches_SE = compute_n_examples(num_requests_SE, 20) // batch_size
num_test_SE = num_test_batches_SE * batch_size

num_train_batches_WIKI = compute_n_examples(num_requests_WIKI, 70) // batch_size
num_train_WIKI = num_train_batches_WIKI * batch_size
num_train_batches_SE = compute_n_examples(num_requests_SE, 70) // batch_size
num_train_SE = num_train_batches_SE * batch_size

requests_test_WIKI = requests_WIKI[:num_test_WIKI]
labels_test_WIKI = labels_WIKI[:num_test_WIKI]
requests_test_SE = requests_SE[:num_test_SE]
labels_test_SE = labels_SE[:num_test_SE]

requests_train_WIKI = requests_WIKI[(-num_train_WIKI):]
labels_train_WIKI = labels_WIKI[(-num_train_WIKI):]
requests_train_SE = requests_SE[(-num_train_SE):]
labels_train_SE = labels_SE[(-num_train_SE):]

# Speicify configurations of GPU
def gpu_config():
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'    
    return config


"""
Copied from: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py

Calculate the average gradient for each shared variable across all towers.
Note that this function provides a synchronization point across all towers.
Args:
    tower_grads: List of lists of (gradient, variable) tuples. 
        The outer list is over individual gradients. 
        The inner list is over the gradient calculation for each tower.
Returns:
    List of pairs of (gradient, variable) where the gradient has been 
    averaged across all towers.
"""
def average_gradients(tower_grads):
    average_grads = []
    
    counter = 0
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, axis=0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, axis=0)
        
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1] # [0]: first tower [1]: ref to var
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


"""
Compute average gradients, perform gradient clipping and apply gradients
Args:
    tower_grad: gradients collected from all GPUs
Returns:
    the op of apply_gradients
"""

def apply_grads(optimizer, tower_grads):
    # averaging over all gradients
    avg_grads = average_gradients(tower_grads)

    # Perform gradient clipping
    (gradients, variables) = zip(*avg_grads)
    (clipped_gradients, _) = tf.clip_by_global_norm(gradients, clipping_threshold)

    # Apply the gradients to adjust the shared variables.
    apply_gradients_op = optimizer.apply_gradients(zip(clipped_gradients, variables))

    return apply_gradients_op

def lstm(input_size, hidden_size, keep_prob, reuse):
    cell = tf.contrib.rnn.LSTMCell(
        hidden_size, use_peepholes=True, # allow implementation of LSTMP?
        initializer=tf.contrib.layers.xavier_initializer(),
        forget_bias=1.0, reuse=reuse)
    cell_dropout = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob,
        variational_recurrent=True, input_size=input_size, dtype=tf.float32)
    return cell_dropout

def get_keep_prob(dropout_rate, is_training):
    keep_prob = tf.cond(
        is_training, 
        lambda: tf.constant(1.0 - dropout_rate),
        lambda: tf.constant(1.0))
    return keep_prob

graph = tf.Graph()

with graph.as_default():
    with tf.variable_scope("classifier"):
        with tf.device('/cpu:0'):
            # Placeholders
            inputs = tf.placeholder(
                tf.string if string_input else tf.int32, 
                shape=[batch_size, None], name="inputs")
            tf.add_to_collection("placeholder", inputs)
            
            seq_lengths = tf.placeholder(tf.int32, shape=[batch_size], name="seq_lengths")
            tf.add_to_collection("placeholder", seq_lengths)
            max_seq_length = tf.reduce_max(seq_lengths)
            mask = tf.sequence_mask(
                seq_lengths, maxlen=max_seq_length, dtype=tf.float32)

            labels = tf.placeholder(tf.int32, shape=[batch_size], name="labels")
            tf.add_to_collection("placeholder", labels)

            is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
            tf.add_to_collection("placeholder", is_training)
            keep_prob = get_keep_prob(dropout_rate, is_training)
            keep_prob_lstm = get_keep_prob(dropout_rate_lstm, is_training)
            
            if string_input:
                table = tf.contrib.lookup.HashTable(
                    tf.contrib.lookup.KeyValueTensorInitializer(
                        vocab_politeness, list(range(len(vocab_politeness)))), 
                    0)
                inputs_filtered = table.lookup(
                    inputs, name="inputs_filtered")
            else:
                # Filter out tokens that the classifier doesn't know
                vocab_mask = tf.cast(
                    inputs < vocab_size_politeness,
                    tf.int32)
                inputs_filtered = inputs * vocab_mask
            
            with tf.variable_scope("embedding"):
                embedding_unk = tf.get_variable(
                    "embedding_unk",
                    shape=[1, embedding_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                embedding_politeness_new = tf.get_variable(
                    "embedding_politeness_new",
                    shape=[new_vocab_size_politeness, embedding_size], 
                    initializer=tf.contrib.layers.xavier_initializer())
                embedding_politeness_original = tf.get_variable(
                    "embedding_politeness_original",
                    shape=[shared_vocab_size_politeness, embedding_size],
                    initializer=tf.constant_initializer(embedding_word2vec_politeness),
                    trainable=True)
                embedding = tf.concat(
                    [embedding_unk, 
                     embedding_politeness_new, 
                     embedding_politeness_original],
                    axis=0)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            predictions_lst = []
            tower_grads = []
            scores_lst = []
            reuse = False
        for i in xrange(num_gpus):
            with tf.device('/gpu:%d' % i):
                if (i == 1):
                    reuse = True

                start = i * batch_size_per_gpu
                end = start + batch_size_per_gpu

                # Embedding layer
                with tf.variable_scope("embedding", reuse=reuse):
                    embedded_inputs = tf.nn.embedding_lookup(
                        embedding, inputs_filtered[start:end, :])            

                with tf.variable_scope("lstm", reuse=reuse):
                    cell_fw = lstm(
                        embedding_size, hidden_size,
                        keep_prob_lstm, reuse)
                    cell_bw = lstm(
                        embedding_size, hidden_size,
                        keep_prob_lstm, reuse)                   
                    
                    (outputs, final_state) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, embedded_inputs,
                        sequence_length=seq_lengths[start:end],
                        dtype=tf.float32,
                        swap_memory=True)

                # H's shape: batch_size_per_gpu * max_seq_length * (2 * hidden_size)
                H = tf.concat(outputs, axis=2)

                # CNN + maxpooling layer
                with tf.variable_scope("CNN_maxpooling", reuse=reuse):
                    H_expanded = tf.expand_dims(H, axis=-1) # expand H to 4-D (add a chanel dim) for CNN

                    pooled_outputs = []

                    for (j, filter_size) in enumerate(filter_sizes):
                        with tf.variable_scope("filter_%d" % j): # sub-scope inherits the reuse flag
                            # CNN layer
                            filter_shape = [filter_size, 2 * hidden_size, 1, num_filters]

                            W_conv = tf.get_variable(
                                "W_conv", shape=filter_shape,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
                            b_conv = tf.get_variable(
                                "b_conv", shape=[num_filters],
                                initializer=tf.constant_initializer(0.1))
                            conv = tf.nn.conv2d(
                                H_expanded, W_conv, strides=[1, 1, 1, 1], padding="VALID")
                            output_conv = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
                            
                            # maxpooling layer
                            batch_maxpooled = tf.map_fn(
                                lambda x: tf.reduce_max(x[0][:x[1], 0, :], axis=0),
                                (output_conv, seq_lengths), dtype=tf.float32)
                            pooled_outputs.append(batch_maxpooled)
                    h_maxpool = tf.nn.dropout(
                        tf.concat(pooled_outputs, axis=1),
                        keep_prob=keep_prob)
                    
                # output layer (fully connected)
                with tf.variable_scope("output", reuse=reuse):           
                    num_filters_total = num_filters * len(filter_sizes)
                    W_out = tf.get_variable(
                        "W_out", shape=[num_filters_total, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b_out = tf.get_variable(
                        "b_out", shape=[num_classes],
                        initializer=tf.constant_initializer(0.1))
                    logits = tf.nn.xw_plus_b(
                        h_maxpool, weights=W_out, biases=b_out)
                    scores = tf.nn.softmax(logits, dim=-1)
                    scores_lst.append(scores[:, 1])

                # Predictions
                predictions = tf.cast(
                    tf.argmax(logits, axis=1), tf.int32)
                predictions_lst.append(predictions)

                # Loss
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels[start:end]))

                # Compute gradients
                grads = optimizer.compute_gradients(loss)
                valid_grads = [
                    (grad, var) 
                    for (grad, var) in grads 
                    if grad is not None] # the proj vars from LM are None
                tower_grads.append(valid_grads)

        with tf.device('/cpu:0'):
            # Accuracy and Derivative Saliency Information
            batch_scores = tf.concat(scores_lst, axis=0)
            tf.add_to_collection("batch_scores", batch_scores)            
            batch_predictions = tf.concat(predictions_lst, axis=0)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(batch_predictions, labels), "float32"))

            apply_gradients_op = apply_grads(optimizer, tower_grads)
            
            # Initializer
            init = tf.global_variables_initializer()

            saver = tf.train.Saver(tf.trainable_variables())
            
print("Done building graph.")

"""
Pad a batch to max_sequence_length along the second dimension
Args:
    • input_seqs: a list of sequences
    • sequence_lengths: a list of sequence length
Returns:
    • padded
"""

def pad(input_seqs, sequence_lengths):
    max_length = max(sequence_lengths)
    padded = [input_seq + [0] * (max_length - sequence_length) 
              for (input_seq, sequence_length)
              in zip(input_seqs, sequence_lengths)]
    return padded


# In[ ]:


def run(sess, requests, truth_labels, num_batches, mode, epoch, dataset):
    accuracy_lst = []
    for i in xrange(num_batches):
        start = batch_size * i
        end = start + batch_size
        input_seqs = requests[start:end]
        sequence_lengths = [len(request) for request in input_seqs]        
        feed_dict = {
            inputs: pad(input_seqs, sequence_lengths),
            seq_lengths: sequence_lengths,
            labels: truth_labels[start:end]}
        
        if mode == "train":
            feed_dict[is_training] = True
            (train_accuracy, _) = sess.run(
                [accuracy, apply_gradients_op],
                feed_dict=feed_dict)
            accuracy_lst.append(train_accuracy)
            print("Batch %d train accuracy %.1f%%" % (i, (train_accuracy * 100)))
        else:
            feed_dict[is_training] = False
            test_accuracy = sess.run(
                accuracy,
                feed_dict=feed_dict)
            accuracy_lst.append(test_accuracy)

    avg_accuracy = np.mean(accuracy_lst)
    print("Avergate %s accuracy for %s after epoch %d: %.1f%%" % (mode, dataset, epoch, avg_accuracy * 100))   

    if mode == "train":
        saver.save(sess, model_path + "politeness_classifier_%d" % epoch)
        print("Checkpoint %d saved." % epoch)

"""
Score politeness in a file
Args:
    checkpoint: the checkpoint parameters to be restored
    file: the file to be scored
"""
def score(sess, checkpoint, sents, restore=False):
    if restore:
        saver.restore(sess, checkpoint)
        print("Restored pretrained variables.")
    
    num_sents = len(sents)
    num_batches = num_sents // batch_size
    
    n_discarded = 0
    all_scores = []
    for i in xrange(num_batches):
        start = batch_size * i
        end = start + batch_size
        input_seqs = sents[start:end]
        
        sequence_lengths = [len(turn) for turn in input_seqs]
        feed_dict = {
            inputs: pad(input_seqs, sequence_lengths),
            seq_lengths: sequence_lengths,
            is_training: False} 
        try:
            scores = sess.run(batch_scores, feed_dict=feed_dict)
            all_scores.extend(scores)
        except:
            n_discarded += 1
            print("Warning: already discarded %d batches!" % n_discarded)
            continue
        print("Scored batch %d" % i)
    
    print("Average score:", (sum(all_scores) / len(all_scores)))

def assign_scores_to_excel(sess, checkpoint, tokenized_forum_file, forum_file, output_file, token2index, restore=False):
    """
    Assign politeness scores to a pre-tokenized forum posts file and save the results to an Excel file.

    Args:
        sess: The TensorFlow session.
        checkpoint: The path to the model checkpoint.
        tokenized_forum_file: The path to the pre-tokenized forum posts .pkl file.
        forum_file: The path to the original forum file (for reading text and adding scores).
        output_file: The path to the output Excel file with politeness scores.
        token2index: The token to index mapping for converting tokens to indices.
        restore: Whether to restore variables from a checkpoint.
    """

    # Load pre-tokenized forum posts from the .pkl file
    print(f"Loading tokenized forum data from {tokenized_forum_file}")
    with open(tokenized_forum_file, 'rb') as f:
        tokenized_sents = pickle.load(f)

    print(f"Loaded {len(tokenized_sents)} forum posts from {tokenized_forum_file}")

    # Convert tokens to indices
    sents = [[token2index.get(token, token2index["UNK_TOKEN"]) for token in sent] for sent in tokenized_sents]

    # Restore model variables if needed
    if restore:
        saver.restore(sess, checkpoint)
        print("Restored pretrained variables.")

    num_sents = len(sents)
    num_batches = num_sents // batch_size
    leftover = num_sents % batch_size  # This will hold the number of remaining samples in the last smaller batch

    all_scores = []
    n_discarded = 0

    # Process full batches
    for i in range(num_batches):
        start = batch_size * i
        end = start + batch_size
        input_seqs = sents[start:end]
        sequence_lengths = [len(turn) for turn in input_seqs]

        feed_dict = {
            inputs: pad(input_seqs, sequence_lengths),
            seq_lengths: sequence_lengths,
            is_training: False
        }

        try:
            scores = sess.run(batch_scores, feed_dict=feed_dict)
            all_scores.extend(scores)
        except Exception as e:
            n_discarded += 1
            print(f"Warning: discarded batch {n_discarded} due to error: {e}")
            continue
        print(f"Scored batch {i}")

    # Handle any leftover sentences (the last batch with fewer than 32 items)
    if leftover > 0:
        # Pad the last batch to batch_size (32)
        input_seqs = sents[num_batches * batch_size:]
        sequence_lengths = [len(turn) for turn in input_seqs]

        # Pad input sequences to reach batch_size
        while len(input_seqs) < batch_size:
            input_seqs.append([token2index["UNK_TOKEN"]])  # Add padding with "UNK_TOKEN"
            sequence_lengths.append(1)

        feed_dict = {
            inputs: pad(input_seqs, sequence_lengths),
            seq_lengths: sequence_lengths,
            is_training: False
        }

        try:
            scores = sess.run(batch_scores, feed_dict=feed_dict)
            # Only keep the real scores (without the padding)
            all_scores.extend(scores[:leftover])
        except Exception as e:
            n_discarded += 1
            print(f"Warning: discarded last batch due to error: {e}")

    print(f"Average score for the entire dataset: {np.mean(all_scores)}")

    # Append politeness scores to DataFrame
    df = pd.read_excel(forum_file, engine='openpyxl')
    if len(df) == len(all_scores):
        df['politeness_score'] = all_scores  # Ensure scores are matched to sentences
    else:
        print(f"Warning: Length of scores ({len(all_scores)}) does not match DataFrame rows ({len(df)}).")

    # Save DataFrame to output Excel file
    df.to_csv(output_file, index=False)
    print(f"Saved politeness scores to {output_file}")

config = gpu_config()

with tf.Session(graph=graph, config=config) as sess:
    sess.run(init)
    if string_input:
        sess.run(tf.tables_initializer())
    print("Initialized all variables.")
    
    if use_all_data:
        all_requests = requests_WIKI + requests_SE
        all_labels = labels_WIKI + labels_SE
        all_num_batches = len(all_requests) // batch_size
        for i in xrange(num_epochs):
            run(sess, all_requests, all_labels, 
                all_num_batches, "train", i, "ALL")
            all_requests, all_labels = shuffle(all_requests, all_labels)
    else:
        if is_test:
            saver.restore(sess, ckpt)
            print("Restored pretrained variables.")
            # Assign politeness scores to Excel file
            if args.tokenized_file and args.output_file:
                assign_scores_to_excel(
                    sess=sess,
                    checkpoint=ckpt,
                    tokenized_forum_file=args.tokenized_file,
                    forum_file=args.forum_file,
                    output_file=args.output_file,
                    token2index=token2index,
                    restore=False
                )
            run(sess, requests_test_WIKI, labels_test_WIKI,
                num_test_batches_WIKI, "test", 2, "WIKI")
            run(sess, requests_test_SE, labels_test_SE,
                num_test_batches_SE, "test", 2, "SE")
        else:
            for i in xrange(num_epochs):
                run(sess, requests_train_WIKI, labels_train_WIKI,
                    num_train_batches_WIKI, "train", i, "WIKI")
                run(sess, requests_test_WIKI, labels_test_WIKI,
                    num_test_batches_WIKI, "test", i, "WIKI")
                run(sess, requests_test_SE, labels_test_SE,
                    num_test_batches_SE, "test", i, "SE")


