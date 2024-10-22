# Politeness of Educational Forum Posts

Author: YUEN, Chak Shing

Faculty of Information Technology

Monash University

## Project Description

Based on the LSTM-CNN politeness model proposed in 2018 in [Niu's paper](https://arxiv.org/abs/1805.03162), I forked the repository and further modified the code needed to process the Stanford forum post data and to assign politeness score using his model, then further analysis was conducted on the stanford forum post dataset.

# 1. LSTM-CNN Politeness Score

## 1.1 Environment
Python: 3.6.13

TensorFlow: 1.15

## 1.2 Install Command After Setting the environment to 3.6.13
```
pip install tensorflow==1.15
pip install pandas
pip install nltk
conda install gensim
pip install jsonlines
pip install python-Levenshtein
conda install gensim
conda install openpyxl
```

## 1.3 Processing Steps

### 1.3.1 Prerequisites:

(1) Obtain the [Stanford Politeness Corpus](http://www.cs.cornell.edu/~cristian/Politeness_files/Stanford_politeness_corpus.zip), unzip it, and put the folder and files inside under data/

(2) Download the [jar file of the Stanford Postagger](https://nlp.stanford.edu/software/tagger.shtml) and put it under root directory, i.e stanford-postagger-full-2020-11-17/stanford-postagger.jar (required for tokenization)

(3) Download the [pretrained word2vec embeddings binary file](https://drive.google.com/uc?export=download&confirm=wa0J&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM) and put it under root directory, i.e. GoogleNews-vectors-negative300.bin

(4) Obtain the [stanfordMOOCForumPostsSet]([https://datastage.stanford.edu/StanfordMoocPosts](https://github.com/akshayka/edxclassify?tab=readme-ov-file)), and put the folder and files inside under data/

### 1.3.2 Preprocess the politeness data of the 3 datasets
```
python src/basic/read_csv.py
python src/basic/process_requests.py  --forum_file data/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.xlsx [path of the excel file] --forum_text_column "Text" [Text column of the excel file] --tagger_path stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar [jar file of the Stanford Postagger] --word2vec GoogleNews-vectors-negative300.bin [pretrained word2vec embeddings binary file]
```
After the preprocessing, the pkl files should be ready for model training, the Shared vocab size is 8765 while the New vocab size is 339.


### 1.3.3 Train the politeness classifier from scratch
```
python src/model/LSTM-CNN-multi-GPU-new_vocab.py
```
After 3 epochs, the model should get average 82.9% and 70.5% accuracies on the WIKI and SE domains respectively (for comparison to results from previous works, please refer to [Niu's paper](https://arxiv.org/abs/1805.03162)). 
Average score for the entire stanford forum posts dataset is 0.7904447317123413

### 1.3.4 Test the politeness classifier and apply the politeness score to a new csv file
```
python src/model/LSTM-CNN-multi-GPU-new_vocab.py --test --ckpt ckpt/politeness_classifier_2 [path of the checkpoint trained model] --tokenized_file data/stanfordMOOCForumPostsSet/tokenized_forum.pkl [path of the preprocessed excel file] --output_file data/stanfordMOOCForumPostsSet/stanfordMOOCForumPostsSet.csv [path of the new csv file]
```

The best trained model is also uploaded to under ckpt/ for easier reference.
Only the scripts mentioned are modified for the project. Other scripts forked from Niu's repository remains unchanged

# 2 Politeness Classifier using Convo toolkit

## 2.1 Environment
Python: 3.9

## 2.2 Processing Steps
With the politeness score generated from LSTM-CNN model, we may then use the convokit package to extract the politeness classifier value and politeness strategies used for each post. 

For details, please refer to the Politeness.ipynb file and [ConvoKit](https://convokit.cornell.edu/). 

# 3 Analysis

## 3.1 Environment
Python: 3.9

## 3.2 Processing Steps
The dataset could be further analyzed for the paper purpose

For details, please refer to the graph.ipynb file and my final paper submitted

# 4 Citations

Please cite Niu's paper if you appear to use the politeness classifier.

Please cite relevant papers if you appear to use the relevant proportion of [ConvoKit](https://convokit.cornell.edu/). 

To be continued on this section


## 4.1 [Niu's paper](https://arxiv.org/abs/1805.03162)

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
