# Politeness of Educational Forum Posts

Author: YUEN, Chak Shing

Initially forked the Github codes from Tong Niu and Mohit Bansal, modified the code and apply to the new dataset.

## Environment
Python: 3.6.13

TensorFlow: 1.3.0

## Command After Setting the environment to 3.6.13
```
pip install pandas
pip install nltk
conda install gensim
pip install jsonlines
pip install python-Levenshtein
```

## Politeness Classifier

(1) Obtain the [Stanford Politeness Corpus](http://www.cs.cornell.edu/~cristian/Politeness_files/Stanford_politeness_corpus.zip), unzip it, and put the files under data/

(2) Download the [jar file of the Stanford Postagger](https://nlp.stanford.edu/software/tagger.shtml) and put it to main directory, i.e stanford-postagger-full-2020-11-17/stanford-postagger.jar (required for tokenization)

(3) Download the [pretrained word2vec embeddings binary file](https://drive.google.com/uc?export=download&confirm=wa0J&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM) and put it to main directory, i.e. GoogleNews-vectors-negative300.bin

To preprocess the politeness data, please run
```
python src/basic/read_csv.py
python src/basic/process_requests.py --tagger_path stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar [path to Stanford Postagger jar file] --word2vec GoogleNews-vectors-negative300.bin [path to pretrained word2vec bin file]
```

To train the politeness classifier from scratch, please run
```
python3 src/model/LSTM-CNN-multi-GPU-new_vocab.py
```

To test the politeness classifier, please run
```
python3 src/model/LSTM-CNN-multi-GPU-new_vocab.py --test --ckpt [name of the checkpoint]
```
The model should get around 85.0% and 70.2% accuracies on the WIKI and SE domains, respectively (for comparison to results from previous works, please refer to [the paper](https://arxiv.org/abs/1805.03162)). 

You can optionally use our trained model [checkpoint](https://drive.google.com/open?id=1593PqiZFk8O1p7095D-8E6KDvxx6j1qQ) by putting it under ckpt/)

## Polite Dialogue Generation


## Citations

Please cite both my paper and Niu's paper if you appear to use the politeness classifier.



[citing our paper](https://transacl.org/ojs/index.php/tacl/rt/captureCite/1424/310/BibtexCitationPlugin).
```
@article{TACL1424,
	author = {Niu, Tong and Bansal, Mohit},
	title = {Polite Dialogue Generation Without Parallel Data},
	journal = {Transactions of the Association for Computational Linguistics},
	volume = {6},
	year = {2018},
	keywords = {},
	abstract = {Stylistic dialogue response generation, with valuable applications in personality-based conversational agents, is a challenging task because the response needs to be fluent, contextually-relevant, as well as paralinguistically accurate. Moreover, parallel datasets for regular-to-stylistic pairs are usually unavailable. We present three weakly-supervised models that can generate diverse, polite (or rude) dialogue responses without parallel data. Our late fusion model (Fusion) merges the decoder of an encoder-attention-decoder dialogue model with a language model trained on stand-alone polite utterances. Our label-fine-tuning (LFT) model prepends to each source sequence a politeness-score scaled label (predicted by our state-of-the-art politeness classifier) during training, and at test time is able to generate polite, neutral, and rude responses by simply scaling the label embedding by the corresponding score. Our reinforcement learning model (Polite-RL) encourages politeness generation by assigning rewards proportional to the politeness classifier score of the sampled response. We also present two retrieval-based, polite dialogue model baselines. Human evaluation validates that while the Fusion and the retrieval-based models achieve politeness with poorer context-relevance, the LFT and Polite-RL models can produce significantly more polite responses without sacrificing dialogue quality.},
	issn = {2307-387X},
	url = {https://transacl.org/ojs/index.php/tacl/article/view/1424},
	pages = {373--389}
}
```
