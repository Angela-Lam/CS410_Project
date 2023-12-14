# CS410 Project: Categrorical Text Classfication based on keywords
Angela Lam, puiyuyl2@illinois.edu

Tingcong Liu, tl17@illinois.edu

# How to run it:
A comprehensive writeup is avaliable [here](https://tichung.com/blog/2021/20200323_flask/).

## Requirements
```
python >= 3.5
pip install numpy python-igraph scikit-learn tqdm inflect nltk datefinder dateutil
```

## Getting started
```bash
git clone https://github.com/Angela-Lam/CS410_Project
cd CS410_Project
#run the script to produce output:
python 410proj.py --data hkprotest
#run the evaluation :
python eval.py --key_event_file data/hkprotest/output.json --ground_truth data/hkprotest/doc2event_id.txt --eval_top 10
```

# Going through each part
This project can be divided into 6 parts:

## Preprocess data 

This project preprocesses data by constructing many small repositories to store the data.

doc2time: document id and its corresponding date published.

min_t: the eariest time the event happens.

num_t: the total time the event happens.

doc_sents: each document is a list, the element of the list is a sentense.

doc_emb: the embedding of each document by finding the embedding of the top 3 sentense.

vocabulary: word that appears at least 10 times and is not puctuations and stop words.

w2tc: word to date count

w2dc: word to document count.
```bash
 doc2time, min_t, num_t, all_t,doc_sents,doc_emb, vocabulary,w2tc,w2dc, docs= data_processing.process_data(args, config)
```

data_processing.py: this 
