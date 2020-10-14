import numpy as np
import pandas as pd

data = pd.read_csv('sdn.csv')

#preprocessing ,onehotencoding and scaling

for i in range(len(data)):
    if(data['pktrate'][i] < 0):
        data.drop(i,axis = 0,inplace = True)

data = data.dropna()

data['index'] = [i for i in range(len(data))]

data['switch'] = data['switch'].astype(str) 
data['src'] = data['src'].astype(str)
data['dst'] = data['dst'].astype(str)
data['port_no'] = data['port_no'].astype(str)

ordered_data = pd.get_dummies(data,
               columns = ['switch','src','dst','port_no'])
values = ordered_data.values

ordered_columns = np.array(ordered_data.columns)
columns_dict = {string:i for i,string in enumerate(ordered_columns)}

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
unscaled_values = np.concatenate((values[:,2:13],values[:,14:19]),axis = 1)
scaled_values = scaler.fit_transform(unscaled_values)

#data with separated protocols

data_tcp = []
data_icmp = []
data_udp = []

protocols = values[:,13]
values = np.concatenate((values[:,19:21],values[:,21:30],values[:,31:49],
             scaled_values,values[:,50:66],values[:,67:71]),axis = 1)

for i in range(len(data)):

    if(protocols[i] == 'UDP'):
        data_udp.append(values[i])
    if(protocols[i] == 'TCP'):
        data_tcp.append(values[i])  
    if(protocols[i] == 'ICMP'):
        data_icmp.append(values[i])

data_tcp = np.array(data_tcp)
data_udp = np.array(data_udp)
data_icmp = np.array(data_icmp)

labels_tcp = np.array(data_tcp[:,0],dtype = np.float64)
labels_udp = np.array(data_udp[:,0],dtype = np.float64)
labels_icmp = np.array(data_icmp[:,0],dtype = np.float64)

data_tcp = np.array(data_tcp[:,1:],dtype = np.float64)
data_udp = np.array(data_udp[:,1:],dtype = np.float64)
data_icmp = np.array(data_icmp[:,1:],dtype = np.float64)

#data with protocol values

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse = False)
protocol_values = encoder.fit_transform(protocols.reshape(-1,1))
original_data = np.concatenate((values,protocol_values[:,:2]),axis = 1)

labels = original_data[:,0]
original_data = original_data[:,1:]

#train test splitting

from sklearn.model_selection import train_test_split

UDP_Xtrain,UDP_Xtest,UDP_Ytrain,UDP_Ytest = train_test_split(data_udp,labels_udp,test_size = 0.1)

TCP_Xtrain,TCP_Xtest,TCP_Ytrain,TCP_Ytest = train_test_split(data_tcp,labels_tcp,test_size = 0.1)

ICMP_Xtrain,ICMP_Xtest,ICMP_Ytrain,ICMP_Ytest = train_test_split(data_icmp,labels_icmp,test_size = 0.1)

X_train = []
X_test = []
Y_train = []
Y_test = []

for i in np.concatenate((UDP_Xtrain[:,0],TCP_Xtrain[:,0],ICMP_Xtrain[:,0])) :
    
    X_train.append(original_data[int(i)])
    Y_train.append(labels[int(i)])
    
for i in np.concatenate((UDP_Xtest[:,0],TCP_Xtest[:,0],ICMP_Xtest[:,0])) :
    
    X_test.append(original_data[int(i)])
    Y_test.append(labels[int(i)])    
    
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
  
#data with separated indices

UDP_train_index,UDP_test_index = UDP_Xtrain[:,0],UDP_Xtest[:,0] 
UDP_Xtrain,UDP_Xtest = UDP_Xtrain[:,1:],UDP_Xtest[:,1:]

TCP_train_index,TCP_test_index = TCP_Xtrain[:,0],TCP_Xtest[:,0] 
TCP_Xtrain,TCP_Xtest = TCP_Xtrain[:,1:],TCP_Xtest[:,1:]

ICMP_train_index,ICMP_test_index = ICMP_Xtrain[:,0],ICMP_Xtest[:,0] 
ICMP_Xtrain,ICMP_Xtest = ICMP_Xtrain[:,1:],ICMP_Xtest[:,1:]

UDP_train_index = np.array(UDP_train_index,dtype = np.int32)
TCP_train_index = np.array(TCP_train_index,dtype = np.int32)
ICMP_train_index = np.array(ICMP_train_index,dtype = np.int32)

UDP_test_index = np.array(UDP_test_index,dtype = np.int32)
TCP_test_index = np.array(TCP_test_index,dtype = np.int32)
ICMP_test_index = np.array(ICMP_test_index,dtype = np.int32)

train_reverse_dict = {x:i for i,x in enumerate(X_train[:,0])}

test_reverse_dict = {x:i for i,x in enumerate(X_test[:,0])}

#SOM for the dataset with protocol

from minisom import MiniSom

som = MiniSom(x = 30, y = 30, input_len = 66, sigma = 5.0, learning_rate = 0.6)

index_scaler = MinMaxScaler()
index_scaler.fit(X_train[:,0].reshape(-1,1))

scaled_index_train = index_scaler.transform(X_train[:,0].reshape(-1,1))

som_X_train = np.concatenate((scaled_index_train,X_train[:,1:]),axis = 1)
som_X_train = np.array(som_X_train,dtype = np.float64)

som.random_weights_init(som_X_train)
som.train_random(som_X_train,120000)

scaled_index_test = index_scaler.transform(X_test[:,0].reshape(-1,1))

som_X_test = np.concatenate((scaled_index_test,X_test[:,1:]),axis = 1)
som_X_test = np.array(som_X_test,dtype = np.float64)

win_map = som.win_map(som_X_train)

#SVC classification

from sklearn.svm import SVC

#SVC for TCP
classifier_tcp = SVC(C = 2.5,kernel = 'poly')
classifier_tcp.fit(TCP_Xtrain,TCP_Ytrain)

supports_tcp = classifier_tcp.support_
dist_tcp = classifier_tcp.decision_function(TCP_Xtrain[supports_tcp])
min_dist_tcp = np.min(dist_tcp)
max_dist_tcp = np.max(dist_tcp)
"classifier_tcp.score(TCP_Xtest,TCP_Ytest)"

#SVC_SOM for TCP
count_tcp = 0
for i in range(len(TCP_Xtest)):
    
    dist_tcp_ = classifier_tcp.decision_function(TCP_Xtest[i].reshape(-1,63))
    
    if( min_dist_tcp <= dist_tcp_ and dist_tcp_ <= max_dist_tcp ):
        
        index = TCP_test_index[i]
        index = test_reverse_dict[index]
        
        winner = som.winner(som_X_test[index])
        x,y = winner[0],winner[1]
        
        win_list = np.array(win_map[(x,y)])
        
        if(len(win_list) == 0):
            continue 
        
        count = 0
        for j in index_scaler.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(j[0]))
            index_ = train_reverse_dict[index_]
            
            if(Y_train[index_] == 1):
                count += 1
                
        this_label = 0
        if(count > 0.7*len(win_list[:,0])):
            this_label = 1
        
        if(this_label == int(TCP_Ytest[i])):
            count_tcp += 1
            
    else:
        this_label = classifier_tcp.predict(TCP_Xtest[i].reshape(-1,63))[0]
        
        if(this_label == TCP_Ytest[i]):
            count_tcp += 1       

#SVC for UDP
classifier_udp = SVC(C = 2.5,kernel = 'linear')
classifier_udp.fit(UDP_Xtrain,UDP_Ytrain)

supports_udp = classifier_udp.support_
dist_udp = classifier_udp.decision_function(UDP_Xtrain[supports_udp])
min_dist_udp = np.min(dist_udp)
max_dist_udp = np.max(dist_udp)

"classifier_udp.score(UDP_Xtest,UDP_Ytest)"

#SVC_SOM for UDP
count_udp = 0
for i in range(len(UDP_Xtest)):
    
    dist_udp_ = classifier_udp.decision_function(UDP_Xtest[i].reshape(-1,63))
    
    if( min_dist_udp <= dist_udp_ and dist_udp_ <= max_dist_udp ):
        
        index = UDP_test_index[i]
        index = test_reverse_dict[index]
        
        winner = som.winner(som_X_test[index])
        x,y = winner[0],winner[1]
        
        win_list = np.array(win_map[(x,y)])
        
        if(len(win_list) == 0):
            continue 
        
        count = 0
        for j in index_scaler.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(j[0]))
            index_ = train_reverse_dict[index_]
            
            if(Y_train[index_] == 1):
                count += 1
                
        this_label = 0
        if(count > 0.5*len(win_list[:,0])):
            this_label = 1
        
        if(this_label == int(UDP_Ytest[i])):
            count_udp += 1
            
    else:
        this_label = classifier_udp.predict(UDP_Xtest[i].reshape(-1,63))[0]
        
        if(this_label == UDP_Ytest[i]):
            count_udp += 1       
            
#SVC for ICMP
classifier_icmp = SVC(C = 2.5,kernel = 'linear')
classifier_icmp.fit(ICMP_Xtrain,ICMP_Ytrain)

supports_icmp = classifier_icmp.support_
dist_icmp = classifier_icmp.decision_function(ICMP_Xtrain[supports_icmp])
min_dist_icmp = np.min(dist_icmp)
max_dist_icmp = np.max(dist_icmp)

"classifier_icmp.score(ICMP_Xtest,ICMP_Ytest)"

#SVC_SOM for ICMP
count_icmp = 0
for i in range(len(ICMP_Xtest)):
    
    dist_icmp_ = classifier_icmp.decision_function(ICMP_Xtest[i].reshape(-1,63))
    
    if( min_dist_icmp <= dist_icmp_ and dist_icmp_ <= max_dist_icmp ):
        
        index = ICMP_test_index[i]
        index = test_reverse_dict[index]
        
        winner = som.winner(som_X_test[index])
        x,y = winner[0],winner[1]
        
        win_list = np.array(win_map[(x,y)])
        
        if(len(win_list) == 0):
            continue 
        
        count = 0
        for j in index_scaler.inverse_transform(win_list[:,0].reshape(-1,1)):
            
            index_ = np.int32(round(j[0]))
            index_ = train_reverse_dict[index_]
            
            if(Y_train[index_] == 1):
                count += 1
                
        this_label = 0
        if(count > 0.7*len(win_list[:,0])):
            this_label = 1
        
        if(this_label == int(ICMP_Ytest[i])):
            count_icmp += 1
            
    else:
        this_label = classifier_icmp.predict(ICMP_Xtest[i].reshape(-1,63))[0]
        
        if(this_label == ICMP_Ytest[i]):
            count_icmp += 1       
            
#FINAL TEST ACCURACY SCORE

final_score = (count_tcp + count_udp + count_icmp)/len(Y_test)            