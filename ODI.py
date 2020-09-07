#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data=pd.read_csv("C:\\Users\\HARSHITA GUPTA\\Downloads\\datasets_4343_7856_ContinousDataset.csv")
data.head()


# In[3]:


labels = data[['Winner']]
data = data[['Team 1', 'Team 2', 'Ground', 'Host_Country', 'Venue_Team1', 'Venue_Team2', 'Innings_Team1', 'Innings_Team2']]
data.head()


# In[4]:


data_hot = pd.get_dummies(data)
data_hot


# In[7]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels.ravel())
labels = le.transform(labels.ravel())


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_hot, labels, test_size=0.1)


# In[9]:


from sklearn.ensemble import RandomForestClassifier as rfc
clf = rfc(n_estimators=100, max_depth=2, random_state=0)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
preds


# In[14]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,preds)
cm
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,preds)
ac


# In[11]:



from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train, y_train)
preds = classifier.predict(x_test)
preds


# In[12]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,preds)
cm


# In[13]:


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,preds)
ac


# In[ ]:




