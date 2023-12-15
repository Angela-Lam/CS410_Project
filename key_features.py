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

# Define a function for key event feature generation
def generate_key_event_features(peak_phrases, w2dc, doc2time, min_t):
    print('key event feature generation')
    top_times = set([pt[1] for pt in peak_phrases])
    prev = set()
    prev_t = None
    nodes = set()
    edge2weight = {}
    for t in sorted(top_times):
        if prev_t and (t-prev_t).days != 1:
            prev = set() #reset peak phrae if its different in 2 consecutive days
        pt_on_t = [pt for pt in peak_phrases if pt[1]==t] #all peak phrase in the day
        nodes.update(pt_on_t) 
        for pt0, pt1 in combinations(pt_on_t, 2):#loop unique peak phrase
            total = len([tt for tt in doc2time if tt == t]) # num of all docs in the day
            docs1 = set([d for d in w2dc[pt0[0]] if doc2time[d] == t]) #num of docs with the key word in that day
            docs2 = set([d for d in w2dc[pt1[0]] if doc2time[d] == t]) #当
            inter = len(docs1.intersection(docs2)) + 1e-5 #intersection 
            npmi = -np.log(inter * float(total) / len(docs1) / len(docs2)) / np.log(inter / float(total)) #cslculate npmi as describes in the paper
            if '_'+pt0[0]+'_' in '_'+pt1[0]+'_' \
            or '_'+pt1[0]+'_' in '_'+pt0[0]+'_' \
            or pt0[0] == infect_engine.plural(pt1[0]): #if the words incompases eachother，npmi=1
                npmi = 1
            if npmi >= 0:
                edge2weight[(pt0, pt1)] = npmi #weight edges
        for p, t in pt_on_t:
            if p in prev:
                edge2weight[((p,t), (p, t-datetime.timedelta(days=1)))] = 3 #如果两个词是连续关系，weight=3
        prev = set([p for p,t in pt_on_t])
        prev_t = t

    g = ig.Graph()
    nodes = list(nodes)
    n2i = {n:i for i,n in enumerate(nodes)} #put nodes in dict
    g.add_vertices(len(nodes))
    edges = [(n2i[i], n2i[j]) for i,j in edge2weight.keys()] #list of edges
    weights = [edge2weight[(nodes[i], nodes[j])] for i,j in edges]
    g.add_edges(edges)
    levels = g.community_multilevel(weights=weights, return_levels=True) 
    
    p_clusters = []
    for ci, c in enumerate(levels[-1]):
        c = [nodes[i] for i in c]
        if len(c) < 2:
            continue
        c_t2p = defaultdict(list)
        for pt in c:
            c_t2p[pt[1]].append(pt[0])
        cluster = set()
        for t in sorted(c_t2p.keys()):
            for pp in c_t2p[t]:
                cluster.add(pp) #将cluster（time，peak phrase）
        p_clusters.append((cluster, sorted(c_t2p.keys()))) #p_cluster: all peak phrase in cluster, in the order if time 
    p_clusters = sorted(p_clusters, key=lambda x: x[1][0]) #p_cluster in time order 

    events = []
    for di, (cluster, ts) in enumerate(p_clusters):
        tis = [(t-min_t).days for t in ts]
        events.append((list(cluster))) #event: cluster peak phrase，starting and ending from min_t=0
    return events
