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
To run the script, you will need to supply various command-line arguments related to the input data:
to produce output: python 410proj.py --data hkprotest
to evaluate for precision, recall and F1 score: python eval.py     --key_event_file data/hkprotest/output.json     --ground_truth data/hkprotest/doc2event_id.txt     --eval_top 10
```

## Going  through each part
410proj.py: it calls the helper files to generate classification and post-classification processing, and saves results. 
data_processing.py: this 
