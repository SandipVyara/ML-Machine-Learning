#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ## 1. Create a simple text dataset

# In[2]:


df_training_raw_data = pd.DataFrame({'sentence': [
                                        'A great game!', 
                                        'The election was over.', 
                                        'A very clean match.', 
                                        'A clean but forgettable game.', 
                                        'It was a close election.',
                                        'The match was exciting!',
                                        'Are you game?',
                                        'Every single vote counts.',
                                        'Have you voted yet?',
                                        'The election results will be out soon.'], 
                                     'tag': [
                                         'Sports', 
                                         'Non Sports', 
                                         'Sports', 
                                         'Sports', 
                                         'Non Sports',
                                         'Sports',
                                         'Sports',
                                         'Non Sports',
                                         'Non Sports',
                                         'Non Sports']})
df_training_raw_data


# ## 2. Preprocessing

# ### 2.1 Converting the tags to numbers

# In[3]:


df_training_raw_data['num_tag'] = df_training_raw_data.tag.astype('category')
df_training_raw_data.num_tag = df_training_raw_data.num_tag.cat.codes


# ### 2.2 Cleaning the text

# In[4]:


import spacy
import string
list_punctuation = [p for p in string.punctuation]
spacy_nlp = spacy.load('en')

def pre_process_text(str_text):
    sp_text = spacy_nlp(str_text)
    list_filtered_tokens = [token.lemma_ for token in sp_text if ((token.text.lower() not in spacy_nlp.Defaults.stop_words) and (token.text not in list_punctuation))]
    return ' '.join(list_filtered_tokens)


# In[5]:


df_training_raw_data['sentence_cleaned'] = df_training_raw_data.sentence.apply(lambda x: pre_process_text(x))


# In[6]:


df_training_raw_data


# ### 2.3 Text Representation using CountVectorizer() from sklearn

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit(df_training_raw_data.sentence_cleaned.tolist())
df_training_raw_data['sentence_vector'] = count_vectorizer.transform(df_training_raw_data.sentence_cleaned.tolist()).toarray().tolist()


# In[8]:


df_training_raw_data


# In[9]:


count_vectorizer.vocabulary_ # word: index number


# ## 3. Training and testing a GNB model

# In[10]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(df_training_raw_data, df_training_raw_data.num_tag, random_state=0)


# In[12]:


X_train


# In[13]:


X_test


# In[14]:


y_train


# In[15]:


y_test


# In[16]:


gnb_model_1 = GaussianNB()
gnb_model_1.fit(np.array(X_train.sentence_vector.tolist()), y_train)


# In[17]:


y_hat = gnb_model_1.predict(np.array(X_test.sentence_vector.tolist()))
X_test['y_hat'] = y_hat
X_test


# In[18]:


new_sentences = ['A very close game.', 'A new paradigm in elections.', 'Nice election game!']
new_sentences_cleaned = [pre_process_text(ns) for ns in new_sentences]
new_sentences_vector = count_vectorizer.transform(new_sentences_cleaned).toarray()
new_sentences_vector


# In[19]:


gnb_model_1.predict(new_sentences_vector)

