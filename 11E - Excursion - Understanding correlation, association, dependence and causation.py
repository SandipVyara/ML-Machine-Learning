#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import seaborn as sb


# ### Excursion: Understanding correlation, association, dependence and causation!!
# 
# Correlation implies association, but not causation. 
# Conversely, causation implies association, but not correlation.

# #### First a tiny dataset

# In[2]:


rand_norm = np.random.normal(0, 1, 5)
rand_norm


# In[3]:


df_test = pd.DataFrame({'x1': rand_norm})
df_test['x2'] = df_test.x1 ** 2
df_test


# In[4]:


df_test.corr()


# #### Lets try with a bigger dataset 

# In[5]:


rand_uni = random.choices(range(1, 10000000), k=1000000)
rand_uni[:10]


# In[6]:


df_test_2 = pd.DataFrame({'x1': rand_uni})
df_test_2['x2'] = df_test_2.x1 ** 2
df_test_2


# In[7]:


df_test_2.corr()


# #### Lets try another dataset

# In[8]:


rand_norm_2 = np.random.normal(0, 1, 1000000)
rand_norm_2[:10]


# In[9]:


df_test_3 = pd.DataFrame({'x1': rand_norm_2})
df_test_3['x2'] = df_test_3.x1 ** 2
df_test_3


# In[10]:


df_test_3.corr()


# In[11]:


df_test_3['x3'] = abs(df_test_3.x1)
df_test_3


# In[12]:


df_test_3.corr()


# In[13]:


sb.scatterplot(x='x1', y='x2', data=df_test_3)


# In[14]:


sb.scatterplot(x='x3', y='x2', data=df_test_3)

