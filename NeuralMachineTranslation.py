# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:52:40 2020

@author: hi
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd 
import re
import matplotlib.pyplot as plt
import string
from string import digits
from sklearn.model_selection import train_test_split

lines=pd.read_table('fra.txt',names=['english','french','irrelevant'],
                             nrows=72000)
lines=lines.iloc[:,:2]
lines['english']=lines['english'].apply(lambda x : x.lower())
lines['french']=lines['french'].apply(lambda x : x.lower())

exclude=set(string.punctuation)

lines['english']=lines['english'].apply(lambda x : ''.join
     (char for char in x if char not in exclude ))
lines['french']=lines['french'].apply(lambda x : ''.join
     (char for char in x if char not in exclude ))

from keras.preprocessing.text import Tokenizer
import json

def create_tokens(text):
    
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer
#English preprocessing
eng_tokenizer=create_tokens(lines['english'])
output_dict_eng=json.loads(json.dumps(eng_tokenizer.word_counts))

eng_words_=sorted(output_dict_eng,key=output_dict_eng.get,reverse=True)

eng_words=pd.Series([word for word in output_dict_eng.keys()])
eng_counts=pd.Series([num for num in output_dict_eng.values()])

eng_frame=pd.DataFrame({'words':eng_words,'counts':eng_counts})
eng_frame=eng_frame.sort_values(by='counts',ascending=False)
eng_frame['cum_counts']=eng_frame['counts'].cumsum()

'''eng_frame['cum_freq']=eng_frame['cum_counts']/eng_frame['cum_counts'].max()
eng_frame_ = eng_frame[eng_frame['cum_freq']<0.8]
eng_list = eng_frame_['words'].values
'''
eng_list = [eng_frame['words'][i+1] for i in range(len(eng_frame)) 
                    if eng_frame['counts'][i]>6]
#French preprocessing
fr_tokenizer=create_tokens(lines['french'])
output_dict_fr=json.loads(json.dumps(fr_tokenizer.word_counts))

fr_words_=sorted(output_dict_fr,key=output_dict_fr.get,reverse=True)

fr_words=pd.Series([word for word in output_dict_fr.keys()])
fr_counts=pd.Series([num for num in output_dict_fr.values()])

fr_frame=pd.DataFrame({'words':fr_words,'counts':fr_counts})
fr_frame=fr_frame.sort_values(by='counts',ascending=False)
fr_frame['cum_counts']=fr_frame['counts'].cumsum()

'''fr_frame['freq']=fr_frame['cum_counts']/fr_frame['cum_counts'].max()
fr_frame_=fr_frame[fr_frame['freq']<0.805]
fr_list=fr_frame_['words'].values
'''
fr_list = [fr_frame['words'][i+1] for i in range(len(fr_frame)) 
                    if fr_frame['counts'][i]>6]

fr_list.insert(0,'<start>')

fr_list.insert(0,'<end>')

lines['french']=lines['french'].apply(lambda x : '<start> '+x+' <end>')

def filter_eng(x):
    arr=[]
    for word in x.split():
        if word in eng_list:
            arr.append(word)
        else:
            arr.append('<unk>')
    return arr 

def filter_fr(x):
    arr=[]
    for word in x.split():
        if word in fr_list:
            arr.append(word)
        else:
            arr.append('<unk>')  
    return arr     

lines['french']=lines['french'].apply(filter_fr)
lines['english']=lines['english'].apply(filter_eng)

extend_eng=eng_list
extend_fr=fr_list

extend_eng.insert(0,'<unk>')
extend_fr.insert(0,'<unk>')

eng_dict={word:i for i,word in enumerate(extend_eng,1)}
fr_dict={word:i for i,word in enumerate(extend_fr,1)}

lines['eng_len']=lines['english'].apply(lambda x:len(x))
lines['fr_len']=lines['french'].apply(lambda x:len(x))

maxlen_eng=lines['eng_len'].max()
maxlen_fr=lines['fr_len'].max()

#encoder decoder luong attention
from keras.layers import Embedding
from keras.layers import Dense,LSTM
from keras.models import Model
from keras.layers import Input 
from keras.layers import dot,Dropout,Activation
from keras.layers import concatenate

encoder_input_data = np.zeros((len(lines),maxlen_eng),dtype = np.float32)
decoder_input_data = np.zeros((len(lines),maxlen_fr),dtype=np.float32)
decoder_output_data= np.zeros((len(lines),maxlen_fr,len(fr_dict)+1),dtype = np.float32)

for i,(english,french) in enumerate(zip(lines['english'],lines['french'])):
    
    for t,word in enumerate(english):
        encoder_input_data[i,t] = eng_dict[word]
        
    for t,word in enumerate(french):
        decoder_input_data[i,t] = fr_dict[word]
        
        if t>0:
            decoder_output_data[i,t-1,fr_dict[word]] = 1
            
        if t == len(french) - 1:
            decoder_output_data[i,t:,2] = 1

for i in range(len(lines)):
    for j in range(maxlen_fr):
        if(decoder_input_data[i][j] == 0 ):
            decoder_input_data[i][j] = 2

embedding_size = 128

encoder_inputs = Input(shape=(maxlen_eng,))
en_x= Embedding(len(eng_dict)+1, embedding_size)(encoder_inputs)
en_x = Dropout(0.1)(en_x)
encoder = LSTM(256, return_sequences=True, unroll=True)(en_x)
encoder_last = encoder[:,-1,:]
            
decoder_inputs = Input(shape=(maxlen_fr,))
dex= Embedding(len(fr_dict)+1, embedding_size)
decoder= dex(decoder_inputs)
decoder = Dropout(0.1)(decoder)
decoder = LSTM(256, return_sequences=True, unroll=True)(decoder,
              initial_state=[encoder_last, encoder_last])

activation_fr = Dense(5000, activation = 'tanh')(decoder)
activation_eng = Dense(5000, activation = 'tanh')(encoder)

attention = dot([activation_fr,activation_eng],axes = [2,2])
attention = Dense(maxlen_eng, activation='tanh')(attention)
attention = Activation('softmax')(attention)

context = dot([attention, encoder], axes = [2,1])

decoder_context = concatenate([context, decoder])
decoder_context=Dense(2000, activation='tanh')(decoder_context)
output=(Dense(len(fr_dict)+1, activation="softmax"))(decoder_context)

model3 = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[output])
model3.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics = ['accuracy'])

model3.summary()

model3.fit([encoder_input_data,decoder_input_data],decoder_output_data,
          batch_size=32,epochs=5,validation_split=0.05)

#prediction by hand
final_predictions = []
for i in range(5000):
    word = 3
    decoder_predictions = np.zeros(maxlen_fr,dtype = 'float32')
    for j in range(maxlen_fr):
        decoder_predictions[j] = word
        pred = model3.predict([encoder_input_data[67000+i].reshape(1,maxlen_eng),
                              decoder_predictions.reshape(1,maxlen_fr)])
        t = np.argmax(pred[0][j])
        word = t
        if word == 2:
            break
    final_predictions.append(decoder_predictions)
            
final_predictions = np.array(final_predictions)

count = 0
correct_count = 0
for i in range(5000):
    correct_count += np.sum((decoder_input_data[67000+i]==final_predictions[i])&
                            (decoder_input_data[67000+i]!=2))
    count += np.sum(decoder_input_data[67000+i]!=2)
    
print(correct_count/count) 
   
# bleu score calculation
from nltk.translate.bleu_score import corpus_bleu

final_predictions = []
for i in range(5000):
    word = 3
    decoder_predictions = np.zeros(maxlen_fr,dtype = 'float32')
    for j in range(maxlen_fr):
        decoder_predictions[j] = word
        pred = model3.predict([encoder_input_data[67000+i].reshape(1,maxlen_eng),
                              decoder_predictions.reshape(1,maxlen_fr)])
        t = np.argmax(pred[0][j])
        word = t
        if word == 2:
            break
    final_predictions.append(decoder_predictions)

references = []
candidates = []

reverse_frdict = {i:word for word,i in fr_dict.items()}

for line in final_predictions:
    target = []
    for i in range(len(line)):
        if int(line[i]) == 0 :
            continue
        target.append(reverse_frdict[line[i]])
        if int(line[i]) == 2 :
            break
    candidates.append(target)   
    
for line in lines['french'][67000:]:
    references.append([line])
print(corpus_bleu(references,candidates,weights=(1,0,0,0)))  
print(corpus_bleu(references,candidates,weights=(0.5,0.5,0,0)))
print(corpus_bleu(references,candidates,weights=(0.33,0.33,0.33,0)))
print(corpus_bleu(references,candidates,weights=(0.25,0.25,0.25,0.25)))
      