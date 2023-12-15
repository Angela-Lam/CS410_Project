import os
import argparse
from utils import *
from collections import defaultdict
import numpy as np
import igraph as ig
from sklearn.svm import SVC
import re
import json
import sys
import nltk 
nltk.download('stopwords')
from tqdm import tqdm
from itertools import product, combinations

import inflect
infect_engine = inflect.engine()

from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english')) 

import datefinder
import dateutil
MONTHS = dateutil.parser.parserinfo.MONTHS
# Define a function for data processing
def process_data(args, config):
        #doc2time: match each event to their corresponding dates. index 0-N
    #min_t: ealiest time 
    #num_t: duration of event
    doc2time, min_t, num_t, all_t = load_doc_time(os.path.join('data', args.data, args.doc_time))
    #docs:each document
    #doc_sents: each document is a list, the element of the list is a sentense.
    docs, doc_sents = load_ucphrase(os.path.join('data', args.data, args.ucphrase_res), len(doc2time))

    #the embedding of each document by finding the embedding of the top 3 sentense
    doc_emb = np.load(os.path.join('data', args.data, args.doc_emb))

    # Construct initial vocab
    word_count = defaultdict(int)
    for doc in docs:
        words = doc.split(' ')
        for word in words:
            word_count[word] += 1
    #vocabulary: word that appears at least 10 times and is not puctuations and stop words.
    vocabulary = [w for w in word_count if word_count[w] >= 10 and w not in stop_words and w not in string.punctuation]

    # word to document count: count of each word in each doc
    w2dc = {w:word_counting(w, docs) for w in tqdm(vocabulary)}

    # filter with lower tf-idf
    w2tfidf = {}
    for w, dc in w2dc.items():
        if len(dc) == 0:
            w2tfidf[w] = 0
        else:
            w2tfidf[w] = np.log(np.sum(list(dc.values()))+1) * np.log(float(len(docs)) / len(dc))
    #print(w2tfidf)
    w_tfidf_num = int(len(w2tfidf) * 0.3)
    w_tfidf_thres = np.partition(list(w2tfidf.values()), kth=w_tfidf_num)[w_tfidf_num]
    #print(w_tfidf_thres)

    #w2tc:document the time each word appears. if this word appears >'phrase_single_day_freq' in a day, and tf-idf higher than shreshold，record when it happens in the day
    w2tc = {w:{} for w in vocabulary}
    for w, dc in w2dc.items():
        for did, c in dc.items():
            if doc2time[did] not in w2tc[w]:
                w2tc[w][doc2time[did]] = []
            w2tc[w][doc2time[did]].append(c)
    w2tc = {w:{t:(np.sum(c) if len(c) > config['phrase_single_day_freq'] and w2tfidf[w] > w_tfidf_thres else 0) for t,c in tc.items()} for w, tc in w2tc.items()}
    #print(w2tc)
    return  doc2time, min_t, num_t, all_t,doc_sents,doc_emb, vocabulary,w2tc,w2dc, docs


