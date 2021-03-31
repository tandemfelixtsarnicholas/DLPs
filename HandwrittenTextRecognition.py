import numpy as np
import matplotlib.pyplot as plt
import cv2
import editdistance as edit

file = open('words.txt')
x=[]
y=[]

numpoints = 0
for line in file:
    
    linesplit = line.strip().split(' ')
    filepath = linesplit[0].split('-')
    imagepath = filepath[0]+'/'+filepath[0]+'-'+filepath[1]+'/'+linesplit[0]+'.png'
    
    imagepath = 'iam-handwritten/' + imagepath      
    img = cv2.imread(imagepath)
    x.append(img)
    imageword = linesplit[-1]
    y.append(imageword)
    
    numpoints += 1
    if(numpoints == 90000):
        break
    
x_new = [] 
y_new = []

for image,label in zip(x,y):
    
    if(len(label) < 15):
        x_new.append(image)
        y_new.append(label)
        
def transform_image(image):
    
    image = np.array(image)
    target=np.ones((32,128),dtype='int8')*255
    shape0=32/image.shape[0]
    shape1=128/image.shape[1]
    
    ratio=min(shape0,shape1)
    x_ = int(image.shape[0]*ratio)
    y_ = int(image.shape[1]*ratio)
    image=cv2.resize(image,(y_,x_))
    target[:x_,:y_]=image[:,:,0]
    target[x_:,y_:]=255
    target = 255 - target
    return target

x=[]
y=[]
for image,word in zip(x_new,y_new):                    
    try:
        x.append(transform_image(image))
        y.append(word)
    except:
        continue
    
import itertools
from tensorflow import keras

listd = y
charList = list(set(list(itertools.chain(*listd)))) 

charList.append(' ')
charList.append('<B>')

char_dict = {x:i for i,x in enumerate(charList)}
reverse_char_dict = {i:x for i,x in enumerate(charList)}
num_images = len(y)
left_images = num_images - 80000
input_lengths = np.ones((num_images,1))*32
label_lengths = np.zeros((num_images,1))
label_indices=[]

for i in range(num_images):
     
     val = []
     for j in y[i]:
         val.append(char_dict[j])
         
     while len(val)<32:
         val.append(79)
     label_indices.append(val)
     label_lengths[i] = len(y[i])
     
x = np.array(x[:num_images]) 
x = x.reshape(num_images,32,128,1)
label_indices = np.array(label_indices)

from keras.backend import ctc_batch_cost

def ctc_loss(args):
     y_pred, labels, input_length, label_length = args
     return ctc_batch_cost(labels, y_pred, input_length, label_length)

from keras.layers import Input
from keras.layers import Conv2D,Activation
from keras.layers import MaxPooling2D 
from keras.layers import add,concatenate 
from keras.layers import Reshape
from keras.layers import GRU
from keras.layers import Lambda
from keras.layers import TimeDistributed
from keras.layers import Dense

input_data = Input(name='inputs', shape = (32, 128,1), dtype='float32')

inner = Conv2D(32, (3,3), padding='same')(input_data)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2,2))(inner)
inner = Conv2D(64, (3,3), padding='same')(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2,2))(inner)
inner = Conv2D(128, (3,3), padding='same')(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2,2))(inner)
inner = Conv2D(128, (3,3), padding='same')(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2,2))(inner)
inner = Conv2D(256, (3,3), padding='same')(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(4,2))(inner)
inner = Reshape(target_shape = ((32,256)))(inner) 

gru_1 = GRU(256,return_sequences = True)(inner)
gru_2 = GRU(256,go_backwards = True,return_sequences = True)(inner)
merged = add([gru_1,gru_2])
gru_3 = GRU(256,return_sequences = True)(merged)
gru_4 = GRU(256,go_backwards = True,return_sequences = True)(merged)

recurrent = concatenate([gru_3,gru_4])
y_pred = TimeDistributed(Dense(80,activation = 'softmax'))(recurrent)

from keras.optimizers import Adam

Optimizer = Adam(lr = 0.002)
labels = Input(name = 'labels', shape=(32,), dtype='float32')
input_length = Input(name='input_length', shape=(1,),dtype='int64')
label_length = Input(name='label_length',shape=(1,),dtype='int64')
output = Lambda(ctc_loss, output_shape=(1,),name='ctc')(
        [y_pred, labels, input_length, label_length])

from keras.models import Model
model = Model(inputs = [input_data, labels, input_length,
                        label_length], outputs = output)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
              optimizer = Optimizer)
model.summary()

x = 1-x/255
x = np.array(x)  

train_input_dict = { 
                     'inputs' : x[:num_images-left_images] ,
                     'labels' : label_indices[:num_images-left_images] ,
                     'input_length' : input_lengths[:num_images-left_images] ,
                     'label_length' : label_lengths[:num_images-left_images]
                   }
train_output_dict = {'ctc' : np.zeros(num_images-left_images)}

from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint(filepath = 'handwriting.hdf5',verbose = 1,
                            save_best_only = True,save_weights_only = True)

model.fit(x = train_input_dict,y = train_output_dict,epochs = 10,verbose = 1,
          batch_size = 32,validation_split = 0.05,callbacks = [model_checkpoint])

model.load_weights('./handwriting.hdf5')
evaluator = Model(inputs = input_data,outputs = y_pred)

from keras.backend import ctc_decode
result = evaluator.predict(x[num_images-left_images:])

input_lengths = input_lengths.reshape(num_images)
prediction = ctc_decode(y_pred = result,input_length = input_lengths[num_images-left_images:],
               greedy = False,beam_width = 16,top_paths = 1)

total_length = 0
error = 0

for i in range(left_images):
    
    actual = []
    predicted = []
    
    for j in range(len(prediction[0][0][i])):
        
        if(prediction[0][0][i][j].numpy() == -1):
            break
        predicted.append(prediction[0][0][i][j].numpy())
    
    for j in range(len(label_indices[num_images-left_images+i])):
        
        if(label_indices[num_images-left_images+i][j] == 79):
            break
        actual.append(label_indices[num_images-left_images+i][j])    
    
    error += edit.distance(actual,predicted)
    total_length += len(actual)     
    
print(error/total_length)    
