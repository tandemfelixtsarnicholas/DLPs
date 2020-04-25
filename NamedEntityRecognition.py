# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:11:31 2020

@author: hi
"""

import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras

unique_words = set()
tags = set()

def splitsentence(filename):
    
    sentences = []
    
    with open(filename,'r') as f :
        next(f)
        next(f)
        sentence = [[],[]]
        
        for line in f:
            splitline = line.split()
            
            if(len(splitline) == 4):
                sentence[0].append(splitline[0])
                sentence[1].append(splitline[-1])
                unique_words.add(splitline[0])
                tags.add(splitline[-1])
            elif(len(splitline) == 0):
                sentences.append(sentence)
                sentence = [[],[]]
            else :
                sentence = [[],[]]
                
        f.close()
    return sentences   

train_data = splitsentence('train.txt')
test_data = splitsentence('test.txt')
valid_data = splitsentence('valid.txt')

chars = set(char for word in unique_words for char in word)

words_index = {w:i+2 for i,w in enumerate(unique_words)}
words_index['UNK'] = 1
words_index['PAD'] = 0

rev_words_index = {i:w for w,i in words_index.items()}

tags_index = {t:i+1 for i,t in enumerate(tags)}
tags_index['PAD'] = 0

rev_tags_index = {i:t for t,i in tags_index.items()}

chars_index = {c:i+2 for i,c in enumerate(chars)}
chars_index['UNK'] = 1
chars_index['PAD'] = 0

rev_chars_index = {i:c for c,i in chars_index.items()}

X_train_word = [[words_index[word] for word in x[0]]for x in train_data]
X_test_word = [[words_index[word] for word in x[0]] for x in test_data]
X_valid_word = [[words_index[word] for word in x[0]] for x in valid_data]

maxlen = 80 

from keras.preprocessing.sequence import pad_sequences

X_train_word = pad_sequences(maxlen = maxlen ,sequences = X_train_word,
                             value = 0,padding = 'post',truncating = 'post')
X_test_word = pad_sequences(maxlen = maxlen ,sequences = X_test_word,
                             value = 0,padding = 'post',truncating = 'post')
X_valid_word = pad_sequences(maxlen = maxlen ,sequences = X_valid_word,
                             value = 0,padding = 'post',truncating = 'post')

y_train = [[tags_index[tag] for tag in x[1]] for x in train_data]
y_test = [[tags_index[tag] for tag in x[1]] for x in test_data]
y_valid = [[tags_index[tag] for tag in x[1]] for x in valid_data]

y_train = pad_sequences(maxlen = maxlen , sequences = y_train , value = 0,
                        padding = 'post' ,truncating = 'post')
y_test = pad_sequences(maxlen = maxlen , sequences = y_test , value = 0,
                        padding = 'post' ,truncating = 'post')
y_valid = pad_sequences(maxlen = maxlen , sequences = y_valid , value = 0,
                        padding = 'post' ,truncating = 'post')

y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],1)
y_test = y_test.reshape(y_test.shape[0],y_test.shape[1],1)
y_valid = y_valid.reshape(y_valid.shape[0],y_valid.shape[1],1)

max_charlen = 16

def splitwords(sentences):
    
    data = []
    for x in sentences:
        sentence = []
        for i in range(maxlen):
            word = []
            for j in range(max_charlen):
                try:
                    word.append(chars_index[x[0][i][j]])
                except:
                    word.append(chars_index['PAD'])
            sentence.append(word)    
        data.append(sentence)    
    return np.array(data)  

X_train_char = splitwords(train_data)
X_test_char = splitwords(test_data)
X_valid_char = splitwords(valid_data) 

from keras.models import Model
from keras.layers import Input,LSTM,Embedding,Dense,TimeDistributed
from keras.layers import Bidirectional,concatenate

word_input = Input(shape = (maxlen,))
word_embedding = Embedding(input_dim = len(words_index),output_dim = 50,
                           input_length = maxlen,mask_zero = True)(word_input)

char_input = Input(shape = (maxlen,max_charlen,))
char_embedding = TimeDistributed(Embedding(input_dim = len(chars_index),output_dim = 10,
                           input_length = max_charlen,mask_zero = True))(char_input)
char_embedding = TimeDistributed(Bidirectional(LSTM(25,return_sequences = False,
                           recurrent_dropout = 0.3)))(char_embedding)

joint_embedding = concatenate([word_embedding,char_embedding])
lstm = Bidirectional(LSTM(100,return_sequences = True,recurrent_dropout = 0.4))(joint_embedding)
output = TimeDistributed(Dense(units = len(tags_index),activation = 'softmax'))(lstm)

model = Model(inputs = [word_input,char_input],outputs = output)

model.summary()

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit([X_train_word,X_train_char],y_train,batch_size = 32,epochs = 10,verbose = 1,
                  validation_data = ([X_valid_word,X_valid_char],y_valid) )