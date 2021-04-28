#!/usr/bin/env python
# coding: utf-8

# In[7]:


import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


# In[12]:


quandl.ApiConfig.api_key = '6fb_DhX1UYTr4vAoFzv-'

df = quandl.get("WIKI/AMZN")
df = df[['Adj. Close']]
df


# In[11]:


df['Adj. Close'].plot(figsize=(15,6), color='g')
plt.legend(loc='upper left')
plt.show


# In[21]:


forecast = 30
df['Prediction'] = df[['Adj. Close']].shift(-forecast)
df

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X)
X_forecast = X[-forecast:]
X = X[:-forecast]

y = np.array(df['Prediction'])
y = y[:-forecast]


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)


forecast_predicted = clf.predict(X_forecast)
print(forecast_predicted)


# In[23]:


plt.plot(X,y)


# In[31]:


dates = pd.date_range(start ="2018-03-28", end = "2018-04-26")
plt.plot(dates, forecast_predicted, color='b')
df['Adj. Close'].plot(color='g')
plt.xlim(xmin=datetime.date(2017,4,26))


# In[ ]:




