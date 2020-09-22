import numpy as np 
import matplotlib.pyplot as plt
import random
import cv2

image_array = []
targets = []

for i in range(1,41):
    for j in range(1,11):
       string = 'AT&T/s'+str(i)+'/'+str(j)+'.PGM'
       image = cv2.imread(string ,cv2.IMREAD_COLOR)
       image = cv2.resize(image,dsize = (224,224),
                          interpolation = cv2.INTER_NEAREST)
       image_array.append(image/255)   
       targets.append(i-1)
    
train_array = np.array(image_array[:350])
y_train = np.array(targets[:350])
test_array = np.array(image_array[350:])
y_test = np.array(targets[350:])

from VGGFace import VGGFaceDescriptor
descriptor = VGGFaceDescriptor((224,224,3))

train_features = descriptor.predict(train_array)
test_features = descriptor.predict(test_array)

from keras.models import Sequential
from keras.layers import Dense

def shared_network(input_shape):
    
    model = Sequential()
    model.add(Dense(1024,activation = 'selu',input_shape = (input_shape,)))
    model.add(Dense(512,activation = 'sigmoid'))
    
    return model

input_shape = train_features.shape[1]
shared_model = shared_network(input_shape)

from keras.layers import Input

first_input = Input((input_shape,))
second_input = Input((input_shape,))

first_output = shared_model(first_input)
second_output = shared_model(second_input)

import keras.backend as K

def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

from keras.layers import Lambda

distance = Lambda(euclidean_distance ,output_shape =(1,))(
        [first_output,second_output])

from keras.models import Model
model = Model(inputs = [first_input,second_input] , outputs = distance)

model.summary()

#trainig data 
pairs,labels = [],[]
class_id = [np.where(y_train == i)[0] for i in range(35)]
    
for i in range(35):
    for j in range(9):
        img1 = train_features[class_id[i][j]]
        img2 = train_features[class_id[i][j+1]]
        pairs.append((img1,img2))
        labels.append(1)
            
        population = list(range(35))
        population.remove(i)
        i_ = random.sample(population,1)[0]
        img1 = train_features[class_id[i][j]]
        img2 = train_features[class_id[i_][j]]
        pairs.append((img1,img2))
        labels.append(0)
        
train_pairs = np.array(pairs)
y_train = np.array(labels)        

#test data        
pairs,labels = [],[]
class_id = [np.where(y_test == i)[0] for i in range(35,40)]

for i in range(5):
    for j in range(9):     
        img1 = test_features[class_id[i][j]]
        img2 = test_features[class_id[i][j+1]]
        pairs.append((img1,img2))
        labels.append(1)
            
        population = list(range(5))
        population.remove(i)
        i_ = random.sample(population,1)[0]
        img1 = test_features[class_id[i][j]]
        img2 = test_features[class_id[i_][j]]
        pairs.append((img1,img2))
        labels.append(0)             
        
test_pairs = np.array(pairs)
y_test = np.array(labels)        

def contrastive_loss(y_true,D):
    margin = 1
    return K.mean(y_true*K.square(D) + (1-y_true)*K.maximum(margin-D,0))

model.compile(loss = contrastive_loss , optimizer = 'adam')
model.fit([train_pairs[:,0],train_pairs[:,1]],y_train,batch_size=64,epochs=10)

result = model.predict([test_pairs[:,0],test_pairs[:,1]])
predictions = [result[i][0] for i in range(len(result))]

count = 0
for x,y in zip(predictions,y_test):
    #print(x,y)
    if(( x<0.44 and y==1 )or( x>=0.44 and y==0 )):
        count = count + 1

print(count/len(y_test)*100)