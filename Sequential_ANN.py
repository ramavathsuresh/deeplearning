#!/usr/bin/env python
# coding: utf-8

# # Generating Dataset

# In[1]:


import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


# In[2]:


train_labels = []
train_samples = []


# Example data:
#     . An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial
#     . The trial had 2100 participants. Half were under 65 years old, half were 65 years or older.
#     . Around 95% of patients 65 0r older experienced side effects.
#     . Around 95% of patients under 65 experienced no side effects.

# In[3]:


for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)


# In[4]:


for i in train_samples:
    print(i)


# In[5]:


for i in train_labels:
    print(i)


# In[10]:


train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)


# In[11]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))


# In[12]:


for i in scaled_train_samples:
    print(i)


# # Creating Model

# In[15]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


# In[19]:


model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])


# In[20]:


model.summary()


# # Training Model

# In[21]:


model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[22]:


model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2)


# In[ ]:




