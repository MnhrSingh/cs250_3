from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)
def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))
  return left, right, Y
test = get_data('snli_1.0_test.jsonl')
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=8)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])
test = prepare_data(test)
from keras.models import load_model
model = load_model('model')
res= model.predict([test[0], test[1]])
file=open('deep_model.txt','a')
for i,j,k in res:
  mx=max(i,j,k)
  if mx==i:
    file.write('contradiction\n')
  elif mx==j:
    file.write('neutral\n')
  else:
    file.write('entailment\n')
file.close()
import pandas as pd
import nltk  
nltk.download('stopwords')
import re
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer  
with open('modellr.pkl', 'rb') as file:
    clf = pickle.load(file)

df=pd.read_json("snli_1.0_test.jsonl",lines=True)
# print(df.iloc[1])
df=df[['sentence1','sentence2','gold_label']]
# print(df)
X=(df['sentence1']+' '+df['sentence2']).tolist()
y=(df['gold_label']).tolist()
processed_sents = []
 
for sent in range(0, len(X)):  
    # Remove all the special characters
    processed_sent = re.sub(r'\W', ' ', str(X[sent]))
 
    # remove all single characters
    processed_sent = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_sent)
 
    # Remove single characters from the start
    processed_sent = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_sent) 
 
    # Substituting multiple spaces with single space
    processed_sent= re.sub(r'\s+', ' ', processed_sent, flags=re.I)
 
    # Removing prefixed 'b'
    processed_sent = re.sub(r'^b\s+', '', processed_sent)
 
    # Converting to Lowercase
    processed_sent = processed_sent.lower()
 
    processed_sents.append(processed_sent)
tfidfconverter = TfidfVectorizer(max_features=500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(processed_sents).toarray()
res=clf.predict(X)
file=open('tfidf.txt','a')
for i in res:
  if 0==i:
    file.write('contradiction\n')
  elif 1==j:
    file.write('neutral\n')
  else:
    file.write('entailment\n')
file.close()