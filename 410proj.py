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

import data_processing
import key_features

def main(args, config):
    # step 1: data processing, see data_processing.py
    doc2time, min_t, num_t, all_t,doc_sents,doc_emb, vocabulary,w2tc,w2dc, docs= data_processing.process_data(args, config)

    # step 2: Peak Phrase Detection
    #print('peak phrase detection')
    wt2score = {}
    for w, t in tqdm(product(vocabulary, all_t)):
        wt2score[(w,t)] = tf_itf(w, t, w2tc, num_t, window_size=3)[0]
    # print("w,t:",len(vocabulary),len(all_t))
    # print(len(wt2score))
    peak_phrases = []

    #sort from high to lowl onlu put into peak phrase if it appears more than 2 times. stop when phrase<500 or score=0
    for pt, s in sorted(wt2score.items(), key=lambda x: x[1], reverse=True):
        if '_' in pt[0]: #consider phrases for more context
            peak_phrases.append(pt)
            if len(peak_phrases) >= 500 or s <= 0:
                break
    #print(peak_phrases)

    #step3 - Key event feature generation
    events = key_features.generate_key_event_features(peak_phrases, w2dc, doc2time, min_t)
    print("this is the event I need")
    for e in events:
        print(e)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='hkprotest')
    parser.add_argument("--ucphrase_res", type=str, default='doc2sents-0.9-tokenized.id.json')
    parser.add_argument("--doc_time", type=str, default='doc2time.txt')
    parser.add_argument("--doc_emb", type=str, default='emb.npy')
    parser.add_argument("--out", type=str, default='output.json')
    args = parser.parse_args()

    print(args.data)
  

    main(args, {'phrase_single_day_freq':3, 'min_pseudo_labels':5})