import tensorflow as tf
import pandas as pd
import numpy as np

train_data = pd.read_csv('RatingsTrain.csv')
train_data = train_data.values
train_data = train_data[:,:3]

max_userid = np.max(train_data[:,0])
max_movieid = np.max(train_data[:,1])

ratings_matrix = np.zeros((max_userid+1,max_movieid+1,5))
rated = set()

for i in range(train_data.shape[0]):
    
    rating = train_data[i][2]
    user = train_data[i][0]
    movie = train_data[i][1]
    
    rated.add((user,movie))
    
    ratings_matrix[user][movie][rating-1] = 1

train_data = ratings_matrix

ranks = 5
batch_size = 32
epochs = 500
display_step = 5
num_hidden = 100
learning_rate = 0.01
k = 2
total_batches = max_userid/batch_size

#global variables
W = tf.Variable(tf.random.normal([(max_movieid+1)*ranks,num_hidden], stddev = 0.01))
b_h = tf.Variable(tf.zeros([1,num_hidden],tf.float32))
b_v = tf.Variable(tf.zeros([1,(max_movieid+1)*ranks],tf.float32))

#helper functions for gibbs sampling
def sample_hidden(probs):
    return tf.floor(probs + tf.random.uniform(tf.shape(probs), 0, 1))

def sample_visible(logits):
    logits = tf.reshape(logits,[-1,ranks])
    sampled_logits = tf.random.categorical(logits,1)             
    logits = tf.one_hot(sampled_logits,depth = 5)
    logits = tf.reshape(logits,[-1,(max_movieid+1)*ranks])
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


def weights_update(xr):
    
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
    
    xr = tf.reshape(x,[-1,(max_movieid+1)*ranks])
    xr = tf.cast(xr,tf.float32)
    h = sample_hidden(tf.sigmoid(tf.matmul(xr,W) + b_h))
    h = tf.cast(h,tf.float32)
    x_ = sample_visible(tf.matmul(h,tf.transpose(W)) + b_v)
    
    logits_pred = tf.reshape(x_,[(max_userid+1),(max_movieid+1),ranks])
    probs = tf.nn.softmax(logits_pred,axis=2)
    return probs

def next_batch():
    while True:
        ix = np.random.choice(np.arange(max_userid+1),batch_size)
        train_X = train_data[ix,:,:]   
        yield train_X
        
for epoch in range(epochs):
    
    if epoch < 150:
        k = 2
    if (epoch > 150) & (epoch < 250):
        k = 3   
    if (epoch > 250) & (epoch < 350):
        k = 5
    if (epoch > 350):
        k = 9      
        
    for i in range(int(total_batches)):
        X_train = next(next_batch())
        X = tf.Variable(X_train)
        X = tf.cast(X,dtype = tf.float32)
        Xr = tf.reshape(X,[-1,(max_movieid+1)*ranks])  
        weights_update(Xr) 
        
    if (epoch % display_step == 0):
        print("Epoch:", '%04d' % (epoch+1))
        
prob_matrix = predict(train_data)

count = 0
square_error = 0

for i in range(max_userid+1):
    for j in range(max_movieid+1):
        
        if(i,j) not in rated:
            continue
        
        predicted = np.argmax(prob_matrix[i,j,:]) + 1
        actual = np.argmax(ratings_matrix[i,j,:]) + 1
        
        square_error += np.power(predicted-actual,2)
        count += 1
        
RMSE = np.sqrt(square_error/count)

print(RMSE)
        