#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


data = pd.read_csv("./train.csv")
data.head()


# In[3]:


print("라벨:", data["Label"].unique(), sep="\n")


# In[ ]:


#3명이 7개의 피쳐에 대해 어떤 분포를 가지고 있는지 시각화
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(data, hue="Label", palette="husl")


# ### preprocess

# In[4]:


import tensorflow as tf
import os
import datetime
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[5]:


train_data = data.fillna(value=0)
user_id_arr = train_data['Label'].unique()
user_count = user_id_arr.shape[0]
train_data = pd.get_dummies(train_data)
train_data = train_data.astype('float64')
train_data = shuffle(train_data)
X_len=len(train_data.columns)-user_count
y_len=user_count
X=train_data.iloc[:,0:X_len]
y=train_data.iloc[:,-user_count:]


# In[6]:


X.head()


# In[7]:


y.head()


# ### train

# In[8]:


#ready model

k=20 #fold
num_val_samples=len(X)//20

sc = StandardScaler()
classifier = Sequential()

unit = 15
rate = 0.3

# 1st layer(input)
classifier.add(Dense(units = unit, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = X_len))

# 2nd layer
classifier.add(Dense(units = unit, kernel_initializer = 'glorot_uniform', activation = 'relu'))
classifier.add(Dropout(rate))

# 3rd layer
classifier.add(Dense(units = unit, kernel_initializer = 'glorot_uniform', activation = 'relu'))
classifier.add(Dropout(rate))

# 4th layer
classifier.add(Dense(units = unit, kernel_initializer = 'glorot_uniform', activation = 'relu'))
classifier.add(Dropout(rate))

# 5th layer(output)
classifier.add(Dense(units = user_count, kernel_initializer = 'glorot_uniform', activation = 'softmax'))

# compile
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()


# In[9]:


#fitting and validation through cross validation

for i in range(k):
    print('fold num #',i+1)
    
    #validation data
    val_data=X[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=y[i*num_val_samples:(i+1)*num_val_samples]
    
    #ready train data and target data
    partial_train_data=np.concatenate([X[:i*num_val_samples], X[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([y[:i*num_val_samples], y[(i+1)*num_val_samples:]],axis=0)
    
    #Feature Scaling
    partial_train_data=sc.fit_transform(partial_train_data)
    val_data=sc.transform(val_data)
    
    #fitting
    classifier.fit(partial_train_data, partial_train_targets, batch_size = 40, epochs = 40)


# ### test - SJW ###

# In[131]:


test_data = pd.read_csv("./test_SJW.csv")
test_data = pd.get_dummies(test_data)
test_data['Label_HYN'] = 0.0
test_data['Label_BYJ'] = 0.0

test_data = test_data.fillna(value=0.0)
test_data = test_data.astype('float64')
test_data = shuffle(test_data)

user_count = 3
X_len=len(test_data.columns)-user_count
y_len=user_count

X=test_data.iloc[:,0:X_len]
y=test_data.iloc[:,-y_len:]

test_pred=classifier.predict(X)
test_pred=test_pred.tolist()
num = len(test_pred[0])
test_pred_df = pd.DataFrame(test_pred,columns=['Label_HYN', 'Label_BYJ', 'Label_SJW'])

print("--------------------------------------------------------")
print("                 BYJ, HYN인 척 하는 SJW                  ")
print("실제로 BYJ:", y["Label_BYJ"].unique(), 
      "실제로 HYN:", y["Label_HYN"].unique(), 
      "실제로 SJW:", y["Label_SJW"].unique(), sep="\n")
print("--------------------------------------------------------")
print("               실제 AI 모델의 판단 결과                 ")
print("BYJ으로 판단:", test_pred_df["Label_BYJ"].unique(), 
      "HYN으로 판단:", test_pred_df["Label_HYN"].unique(), 
      "SJW으로 판단:", test_pred_df["Label_SJW"].unique(), sep="\n")
print("--------------------------------------------------------")

test_pred_df.to_csv('./predict_SJW.csv')


# ### test - HYN ###

# In[132]:


test_data = pd.read_csv("./test_HYN.csv")
test_data = pd.get_dummies(test_data)
test_data['Label_BYJ'] = 0.0
test_data['Label_SJW'] = 0.0

test_data = test_data.fillna(value=0.0)
test_data = test_data.astype('float64')
test_data = shuffle(test_data)

user_count = 3
X_len=len(test_data.columns)-user_count
y_len=user_count

X=test_data.iloc[:,0:X_len]
y=test_data.iloc[:,-y_len:]

test_pred=classifier.predict(X)
test_pred=test_pred.tolist()
num = len(test_pred[0])
test_pred_df = pd.DataFrame(test_pred,columns=['Label_BYJ', 'Label_SJW', 'Label_HYN'])

print("--------------------------------------------------------")
print("                 BYJ, SJW인 척 하는 HYN                  ")
print("실제로 BYJ:", y["Label_BYJ"].unique(), 
      "실제로 HYN:", y["Label_HYN"].unique(), 
      "실제로 SJW:", y["Label_SJW"].unique(), sep="\n")
print("--------------------------------------------------------")
print("               실제 AI 모델의 판단 결과                 ")
print("BYJ으로 판단:", test_pred_df["Label_BYJ"].unique(), 
      "HYN으로 판단:", test_pred_df["Label_HYN"].unique(), 
      "SJW으로 판단:", test_pred_df["Label_SJW"].unique(), sep="\n")
print("--------------------------------------------------------")

test_pred_df.to_csv('./predict_HYN.csv')


# ### test - BYJ ###

# In[134]:


test_data = pd.read_csv("./test_BYJ.csv")
test_data = pd.get_dummies(test_data)
test_data['Label_SJW'] = 0.0
test_data['Label_HYN'] = 0.0

test_data = test_data.fillna(value=0.0)
test_data = test_data.astype('float64')
test_data = shuffle(test_data)

user_count = 3
X_len=len(test_data.columns)-user_count
y_len=user_count

X=test_data.iloc[:,0:X_len]
y=test_data.iloc[:,-y_len:]

test_pred=classifier.predict(X)
test_pred=test_pred.tolist()
num = len(test_pred[0])
test_pred_df = pd.DataFrame(test_pred,columns=['Label_SJW', 'Label_HYN', 'Label_BYJ'])

print("--------------------------------------------------------")
print("                 SJW, HYN인 척 하는 BYJ                  ")
print("실제로 BYJ:", y["Label_BYJ"].unique(), 
      "실제로 HYN:", y["Label_HYN"].unique(), 
      "실제로 SJW:", y["Label_SJW"].unique(), sep="\n")
print("--------------------------------------------------------")
print("               실제 AI 모델의 판단 결과                 ")
print("BYJ으로 판단:", test_pred_df["Label_BYJ"].unique(), 
      "HYN으로 판단:", test_pred_df["Label_HYN"].unique(), 
      "SJW으로 판단:", test_pred_df["Label_SJW"].unique(), sep="\n")
print("--------------------------------------------------------")

test_pred_df.to_csv('./predict_BYJ.csv')
