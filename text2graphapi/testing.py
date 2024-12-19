

from joblib import Parallel, delayed
import os 
import sys
import logging
import time, math
from sklearn.datasets import fetch_20newsgroups

from src.Utils import Utils
from src.Preprocessing import Preprocessing
from src.GraphTransformation import GraphTransformation
from src import Graph
from src import configs



def serial(funct, params):
    data = []
    start = time.time()
    for i in params:
        data.append(funct(i))
    end = time.time()
    print('serial: {:.4f} s'.format(end-start))
    return data


def parallel(funct, params: list):
    data = []
    start = time.time()
    data = Parallel(
        n_jobs=4, 
        backend='loky', 
        mmap_mode='c', 
        max_nbytes=None)(
            delayed(funct)(i) for i in params
        )
    end = time.time()
    print('parallel: {:.4f} s'.format(end-start))
    return data


def handle_20ng_dataset(corpus_docs, num_rows=-1):
    id = 1
    new_corpus_docs = []
    for d in corpus_docs[:num_rows]:
        #doc = {"id": id, "doc": d}
        new_corpus_docs.append(d)
        id += 1
    return new_corpus_docs


def test_function(i):
    time.sleep(1)
    return math.sqrt(i**2)


def text_normalization(text):
    text = prep.handle_blank_spaces(text)
    text = prep.handle_non_ascii(text)
    text = prep.handle_emoticons(text)
    text = prep.handle_html_tags(text)
    text = prep.handle_negations(text)
    text = prep.handle_contractions(text)
    text = prep.handle_stop_words(text)
    text = prep.to_lowercase(text)
    text = prep.handle_blank_spaces(text)
    word_tokenize = prep.word_tokenize(text)
    return text, word_tokenize


# *** BASIC TESTING
params = [1,2,3,4,5,6,7,8,9,10]
#parallel(test_function, params)
#serial(test_function, params)

# *** TEXT NORMALIZATION TESTING
DATASET = '20_newsgroups'
newsgroups_dataset = fetch_20newsgroups() #subset='train'
print(DATASET, len(newsgroups_dataset.data))
corpus_text_docs = handle_20ng_dataset(newsgroups_dataset.data, num_rows=1000)   
prep = Preprocessing(lang='en')

#data = serial(text_normalization, corpus_text_docs)
#data = parallel(text_normalization, corpus_text_docs)
#print(len(data), data[0])

