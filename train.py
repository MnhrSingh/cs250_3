import pandas as pd
df=pd.read_json("snli_1.0_train.jsonl",lines=True)
# print(df.iloc[1])
df=df[['sentence1','sentence2','gold_label']]
# print(df)
X=(df['sentence1']+' '+df['sentence2']).tolist()
y=(df['gold_label']).tolist()
# print(df['sentence1'])
import nltk  
nltk.download('stopwords')
from nltk.corpus import stopwords 
import re
processed_sents = [] 
for sent in range(0, len(X)):  
    processed_sent = re.sub(r'\W', ' ', str(X[sent]))
    processed_sent = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_sent)
    processed_sent = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_sent) 
    processed_sent= re.sub(r'\s+', ' ', processed_sent, flags=re.I)
    processed_sent = re.sub(r'^b\s+', '', processed_sent)
    processed_sent = processed_sent.lower()
    processed_sents.append(processed_sent)
from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(processed_sents).toarray()
pickle.dump(X,open('tfidf.pkl','wb'))
label= {'contradiction': 0, 'neutral': 1, 'entailment': 2}
Y=[]
# print(y[0])
for i in range(len(y)):
    if y[i] in label:
        Y.append(label[y[i]])
    else:
        Y.append(1)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(verbose=1,n_jobs=6)
clf.fit(X,Y)
import pandas as pd
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
    processed_sent = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_sent)
    processed_sent = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_sent) 
    processed_sent= re.sub(r'\s+', ' ', processed_sent, flags=re.I)
    processed_sent = re.sub(r'^b\s+', '', processed_sent)
    processed_sent = processed_sent.lower()
    processed_sents.append(processed_sent)
X = tfidfconverter.transform(processed_sents).toarray()
res=clf.predict(X)
Y=[]
for i in range(len(y)):
    if y[i] in label:
        Y.append(label[y[i]])
    else:
        Y.append(1)
import numpy as np
# print(np.sqrt(sum([(i-j)**2 for i,j in zip(res,Y)])/len(y)))
sum=0
for i in range(len(y)):
    if(Y[i]!=res[i]):
        sum=sum+1
print(sum/len(y))
import pickle
with open('modellr.pkl','wb') as file:
    pickle.dump(clf,file)
# pickle.dump(tfidfconverter,open('tfidf.pkl','wb'))
print(res)
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
    yield (label, s1, s2)\
def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  label = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))
  return left, right, Y
training = get_data('snli_1.0_train.jsonl')
validation = get_data('snli_1.0_dev.jsonl')
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])
VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
RNN = recurrent.GRU
RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))
to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=8)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])
training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)
GLOVE_STORE = 'precomputed_glove.weights'
emb = {}
f = open('glove.840B.300d.txt')
for line in f:
  values = line.split(' ')
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  emb[word] = coefs
f.close()
embm = np.zeros((VOCAB, 300))
for word, i in tokenizer.word_index.items():
  embv = emb.get(word)
  if embv is not None:
    embm[i] = embv
#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
acc=[0.6839,0.7128,0.7315,0.7445,0.7539,0.7600,0.7663,0.7704,0.7744,0.7784,0.7819,0.7846]
val=[0.6544,0.6648,0.6790,0.6569,0.6855,0.6927,0.7041,0.6974,0.6971,0.7069,0.7143,0.7139]
acc=[i*100 for i in acc]
val=[i*100 for i in val]
y=[i for i in range(1,13)]
plt.xlabel('No of Epochs')
plt.ylabel('Accuracy in %')
plt.plot(y,acc,'g',label='Training accuracy')
plt.plot(y,val,'r',label='Validation accuracy')
plt.legend()
plt.savefig('AccuvsEpoch.png')
plt.rcParams["figure.figsize"] = (15,15)
plt.show()
np.save(GLOVE_STORE, embm)
embm = np.load(GLOVE_STORE + '.npy')
embed = Embedding(VOCAB, 300, weights=[embm], input_length=8)
rnn_kwargs = dict(output_dim=300, dropout_W=0.1, dropout_U=0.1)
SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(300, ))
translate = TimeDistributed(Dense(300, activation='relu'))
premise = Input(shape=(8,), dtype='int32')
hypothesis = Input(shape=(8,), dtype='int32')
premis = embed(premise)
hyp = embed(hypothesis)
premis = translate(premis)
hyp = translate(hyp)
rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
premis = rnn(premis)
hyp = rnn(hyp)
premis = BatchNormalization()(premis)
hyp = BatchNormalization()(hyp)
joint = merge([premis, hyp], mode='concat')
joint = Dropout(0.1)(joint)
for i in range(3):
  joint = Dense(2 * 300, activation='relu', W_regularizer=l2(0.000001) )(joint)
  joint = Dropout(0.1)(joint)
  joint = BatchNormalization()(joint)
pred = Dense(len(LABELS), activation='softmax')(joint)
model = Model(input=[premise, hypothesis], output=pred)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
_, tmpfn = tempfile.mkstemp()
model.fit([training[0], training[1]], training[2], batch_size=512, nb_epoch=8, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)
model.load_weights(tmpfn)
loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=512)