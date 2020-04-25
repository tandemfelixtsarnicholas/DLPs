# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:37:03 2020

@author: hi
"""

import tensorflow as tf
import pandas as pd
import numpy as np

data=pd.read_table('ratings.txt')
data.columns=['userId','movieId','rating','timestamp']
users=np.unique(data['userId'].values)
movies=np.unique(data['movieId'].values)

test_data=[]

ratings_matrix=np.zeros([len(users),len(movies),5])

for i in range(len(data)):
    sample=data[i:i+1]
    user_index=int(sample['userId']-1)
    movie_index= int(sample['movieId']-1)
    rating_index=int(sample['rating']-1)
    if np.random.uniform(0,1)<0.2:
        test_data.append([user_index,movie_index,rating_index+1])
    else:
        ratings_matrix[user_index,movie_index,rating_index]=1
    
    
train_data=ratings_matrix

#global constants
num_users=len(users)
num_movies=len(movies)
num_ranks=5
batch_size=32
epochs=500
display_step=5
num_hidden=50
learning_rate=0.01
k=2
total_batches=len(users)/batch_size

#global variables
W = tf.Variable(tf.random.normal([num_movies*num_ranks,num_hidden], 0.01),
                 name="W")
b_h = tf.Variable(tf.zeros([1,num_hidden],tf.float32, name="b_h"))
b_v = tf.Variable(tf.zeros([1,num_movies*num_ranks],tf.float32, name="b_v"))

#helper functions for gibbs sampling
def sample_hidden(probs):
    return tf.floor(probs + tf.random.uniform(tf.shape(probs), 0, 1))

def sample_visible(logits):
    logits = tf.reshape(logits,[-1,num_ranks])
    sampled_logits = tf.random.categorical(logits,1)             
    logits = tf.one_hot(sampled_logits,depth = 5)
    logits = tf.reshape(logits,[-1,num_movies*num_ranks])
    return logits 

#gibbs sampling 
def gibbs_step(x_k):
    h_k = sample_hidden(tf.sigmoid(tf.matmul(x_k,W) + b_h))
    x_k = sample_visible(tf.add(tf.matmul(h_k,tf.transpose(W)),b_v))
    return x_k  
     
def gibbs_sample(k,x_k):
    for i in range(k):
        x_k = gibbs_step(x_k) 
    return x_k 


def gradient_update(xr):
    
    #contrastive divergence
    x_sample = gibbs_sample(k,xr)
    h_sample = sample_hidden(tf.sigmoid(tf.matmul(x_sample,W) + b_h)) 
    
    #hidden sample h_cap
    h = sample_hidden(tf.sigmoid(tf.matmul(xr,W) + b_h)) 
    
    #update
    W_add = tf.multiply(learning_rate/batch_size,
            tf.subtract(tf.matmul(tf.transpose(xr),h),
                        tf.matmul(tf.transpose(x_sample),h_sample)))
    bv_add = tf.multiply(learning_rate/batch_size,
                     tf.reduce_sum(tf.subtract(xr,x_sample), 0, True))
    bh_add = tf.multiply(learning_rate/batch_size, 
                     tf.reduce_sum(tf.subtract(h,h_sample), 0, True))
    W.assign_add(W_add)
    b_v.assign_add(bv_add)
    b_h.assign_add(bh_add)
    
def predict(x):
    
    xr=tf.reshape(x,[-1,num_movies*num_ranks])
    
    h = sample_hidden(tf.sigmoid(tf.matmul(xr,W) + b_h))
    x_ = sample_visible(tf.matmul(h,tf.transpose(W)) + b_v)
    
    logits_pred = tf.reshape(x_,[users,num_movies,num_ranks])
    probs = tf.nn.softmax(logits_pred,axis=2)
    return probs

def next_batch():
    while True:
        ix = np.random.choice(np.arange(num_users),batch_size)
        train_X  = train_data[ix,:,:]   
        yield train_X
        
for epoch in range(epochs):
    
    if epoch < 150:
        k = 2
    if (epoch > 150) & (epoch < 250):
        k = 3   
    if (epoch > 250) & (epoch < 350):
        k = 5
    if (epoch > 350) & (epoch < 500):
        k = 9      
        
    for i in range(int(total_batches)):
        X_train = next(next_batch())
        X = tf.Variable(X_train)
        X = tf.cast(X,dtype=tf.float32)
        Xr = tf.reshape(X,[-1,num_movies*num_ranks])  
        gradient_update(Xr) 
        
    if (epoch % display_step == 0):
        print("Epoch:", '%04d' % (epoch+1))
        
prob_matrix=predict(train_data)
records= []

for i in range(num_users):
    for j in range(num_movies):
        rec = [i,j,np.argmax(prob_matrix[i,j,:]) +1]
        records.append(rec)
            
records = np.array(records)      
        