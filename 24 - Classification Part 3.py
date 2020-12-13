#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_cars_raw_data = pd.read_csv('../data/car.data', header=None, names=['buying', 'maint', 'doors', 'persons', 
                                                                       'lug_boot', 'safety', 'evaluation'])


# ### EDA

# In[3]:


def get_df_info(df, include_unique_values=False):
    col_name_list = list(df.columns)
    col_type_list = [type(df[col][0]) for col in col_name_list]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_memory_usage_list = [df[col].memory_usage(deep=True) for col in col_name_list]
    df_total_memory_usage = sum(col_memory_usage_list) / 1048576
    if include_unique_values:
        col_unique_list = [df[col].unique() for col in col_name_list]
        df_info = pd.DataFrame({'column_name': col_name_list, 'type': col_type_list, 
                                'null_count': col_null_count_list, 'nunique': col_unique_count_list, 
                                'unique_values': col_unique_list})
    else:
        df_info = pd.DataFrame({'column_name': col_name_list, 'type': col_type_list, 
                                'null_count': col_null_count_list, 'nunique': col_unique_count_list})
    return df_info, df_total_memory_usage


# In[4]:


df_cars_raw_data.head()


# In[5]:


df_cars_raw_data_info, df_cars_raw_data_mem_usage = get_df_info(df_cars_raw_data, True)
df_cars_raw_data_info


# In[6]:


for col in df_cars_raw_data:
    print('\nCol {} value_counts: \n{}'.format(col, df_cars_raw_data[col].value_counts()))


# ### Preprocessing

# In[7]:


def get_cat_codes(str_val, col_name):
    
    if str_val in ['low', '2', 'small', 'unacc']:
        return 0
    
    if str_val in ['med', '3', 'acc']:
        return 1
    
    if str_val in ['high', 'more', 'big', 'good']:
        return 2
    
    if str_val in ['vhigh', '5more', 'vgood']:
        return 3
    
    if str_val == '4' and col_name == 'persons':
        return 1
    else:
        return 2


# In[8]:


df_cars = df_cars_raw_data.copy()
for c in df_cars.columns:
    df_cars['x_' + c] = df_cars[c].apply(lambda x: get_cat_codes(x, c))
df_cars = df_cars.iloc[:, 7:]
df_cars.columns = df_cars_raw_data.columns


# ### More EDA

# In[9]:


df_cars.head()


# In[10]:


df_cars_info, df_cars_mem_usage = get_df_info(df_cars, True)
df_cars_info


# In[11]:


df_cars.hist(figsize=(12, 9), layout=(2, 4))


# In[12]:


df_cars.describe()


# In[13]:


import seaborn as sb
sb.set(rc={'figure.figsize': (12, 9)})
sb.heatmap(df_cars.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)


# ### Create and Evaluate Models

# In[14]:


import sklearn
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as NBC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC

from sklearn.model_selection import GridSearchCV, cross_validate

# for metrics, we shall use scorer strings in cross_validate


# In[15]:


sorted(sklearn.metrics.SCORERS.keys())


# In[16]:


model_names = []
model_feature_counts = []
model_feature_names = []
model_acc_scores = []
model_bac_scores = []
model_prc_scores = []
model_rec_scores = []
model_auc_scores = []


# In[17]:


X = df_cars.iloc[:, :-1]


# In[18]:


rfc_model_1_cv = cross_validate(RFC(random_state=0), X, df_cars.evaluation, cv=5, n_jobs=5, verbose=10, scoring=['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'roc_auc_ovr_weighted'])
model_names.append('rfc_model_1_cv')
model_feature_counts.append(X.shape[1])
model_feature_names.append(list(X.columns))
model_acc_scores.append(rfc_model_1_cv['test_accuracy'].mean())
model_bac_scores.append(rfc_model_1_cv['test_balanced_accuracy'].mean())
model_prc_scores.append(rfc_model_1_cv['test_precision_weighted'].mean())
model_rec_scores.append(rfc_model_1_cv['test_recall_weighted'].mean())
model_auc_scores.append(rfc_model_1_cv['test_roc_auc_ovr_weighted'].mean())


# In[19]:


df_model_eval = pd.DataFrame({'model': model_names, 'feature_count': model_feature_counts, 'feature_names': model_feature_names, 
                              'acc': model_acc_scores, 'bac': model_bac_scores, 'prc': model_prc_scores, 'rec': model_rec_scores, 'auc': model_auc_scores})
df_model_eval.round(2)

