# CS410 Project: Sub Event Key Word Detection From Larger Event Corpus
Angela Lam, puiyuyl2@illinois.edu

Tingcong Liu, tl17@illinois.edu

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
```

# Going through each part
This project can be divided into 3 parts:

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

## Peak Phrase Detection

We use our self innovated tf-idf to culculate each word's score.
```bash
wt2score[(w,t)] = tf_itf(w, t, w2tc, num_t, window_size=3)[0]
```
Instead of using normal tf-idf, we got some ideas from EtypeClus to generate salient word:

$$Salient(w) = (1+log(freq(w))^2)log(\frac{N_{bs}}{bsf(w)})$$

After that we sort score from large to small. It stops when score = 0 or the number of phrases >= 500.
```bash
for pt, s in sorted(wt2score.items(), key=lambda x: x[1], reverse=True):
   peak_phrases.append(pt)
```

## Key Event Feature Generation

We use peak phrases we got from last step (peak_phrases),word to document count(w2dc),document id and its corresponding date published(doc2time) and the eariest time the event happens(min_t). We use these as parameters to find events by applying graph algorithm.

```bash
events = key_features.generate_key_event_features(peak_phrases, w2dc, doc2time, min_t)
```
We instead The Modularity Optimization algorithm tries to detect communities in the graph.
```bash
g = ig.Graph()
levels = g.community_multilevel()
```

## Output 
We obtained 2 datasets, one with the background of the Ebola virus, the other with the background of the Hong Kong protests. The model performs evaluation on the corpus to generate sequences of keywords surrounding a key event. 

# On Hong Kong Protest data
Here are some example outputs:
```bash
['legislative_council', 'evil_law', 'victoria_park']
['public_anger', 'young_people', 'civil_human_rights_front', 'prodemocracy_umbrella_movement', 'hong_kong_chief_executive_carrie', 'hong_kongers']
['water_cannon', 'antigovernment_protesters', 'water_cannons', 'gas_masks', 'hong_kong_university', 'university_campus', 'escalating_violence', 'police_officer', 'hong_kong_polytechnic_university']
```

Each list of keywords are descriptions of subevents generated across the documents. For example, the first list describes people ('legislative_council'), object/view point ('evil_law'), and location ('victoria_park'). 


The second descibes protests that's led by civil human rights front (CHRF), a pro-democracy organization based on Hong Kong. Prodemocracy umbrella movement is a political movement characterized by peaceful sitins using umbrellas as a tool. Related to CHRF and the Prodemocracy umbrella movement, are 'public_anger', 'young_people','hong_kong_chief_executive_carrie', 'hong_kongers'. 

The third describes the violence happening in universities in Hong Kong, involving the usage of gas masks and water cannons by the police. 

# On Ebola data
```bash
['public_health', 'medecins_sans', 'secretary_general', 'ebola_crisis', 'security_council']

['ebola_vaccine', 'in_west_africa', 'public_health_agency', 'world_bank']
```
The first one describes the world's response to the Ebola crisis, with involvement from secretary general and security council. The second focuses on efforts to combat the virus in west Africa with involvement from the World Bank, and the development of the vaccine. 
