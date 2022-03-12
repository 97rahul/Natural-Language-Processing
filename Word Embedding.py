# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:56:34 2022

@author: tom97
"""

from keras.preprocessing.text import one_hot

sent=['the glass of milk',
     'the glass of juice',
     'the cup of tea',
     'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good'] 

## One hot encoding
vocab_size = 10000
one_hot_rep = [one_hot(words,vocab_size) for words in sent]


## Word Embedding Representation
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import numpy as np

sent_length = 8
emb_doc = pad_sequences(one_hot_rep,padding = 'post', maxlen = sent_length)


dim = 10

model = Sequential()
model.add(Embedding(vocab_size,dim,input_length = sent_length))
model.compile('adam','mse')


model.summary()

print(model.predict(emb_doc))

print(model.predict(emb_doc[0]))