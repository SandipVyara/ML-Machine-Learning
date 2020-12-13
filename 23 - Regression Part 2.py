#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_boston


# ### Load Data

# In[2]:


boston_dataset = load_boston()


# ### EDA

# In[3]:


type(boston_dataset)


# In[4]:


boston_dataset


# In[5]:


boston_dataset_X = boston_dataset['data']
boston_dataset_Y = boston_dataset['target']
boston_dataset_X_names = boston_dataset['feature_names']
boston_dataset_X_desc = boston_dataset['DESCR']


# In[6]:


print(boston_dataset_X_desc)


# In[7]:


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


# In[8]:


df_boston_raw_data = pd.DataFrame(boston_dataset_X, columns=boston_dataset_X_names)
df_boston_raw_data['MEDV'] = boston_dataset_Y
df_boston_raw_data.head()


# In[9]:


df_boston_raw_data_info, df_boston_raw_data_mem_usage = get_df_info(df_boston_raw_data, True)
df_boston_raw_data_info


# In[10]:


df_boston_raw_data_mem_usage


# In[11]:


import numpy as np


# In[12]:


df_boston_raw_data.CHAS = df_boston_raw_data.CHAS.astype(np.int8)
df_boston_raw_data.RAD = df_boston_raw_data.RAD.astype(np.int8)


# In[13]:


df_boston_raw_data.describe().round(2)


# In[14]:


import seaborn as sb
sb.set(rc={'figure.figsize': (14, 9)})
sb.heatmap(df_boston_raw_data.corr(), cmap='coolwarm', annot=True, vmin=-1, vmax=1)


# ### Create Models and Evaluate

# In[15]:


def show_model_eval_table(model_attrib):
    df_model_eval = pd.DataFrame({'model': model_attrib['model_names'], 'feature_count': model_attrib['model_feature_counts'], 
                                  'feature_names': model_attrib['model_feature_names'], 'r2': model_attrib['model_r2_scores'], 
                                  'mae': model_attrib['model_mae_scores']})
    return df_model_eval.round(2)


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[17]:


from sklearn.metrics import r2_score, mean_absolute_error


# #### Creating X1 with all 13 features

# In[18]:


X1_train, X1_test, y_train, y_test = train_test_split(df_boston_raw_data.iloc[:, :-1], df_boston_raw_data.MEDV, random_state=0)
print(X1_train.shape, y_train.shape)
print(X1_test.shape, y_test.shape)


# #### Creating X2 by dropping 'CHAS' as it has the lowest corr with y

# In[19]:


X2_train = X1_train.drop(columns=['CHAS'])
X2_test = X1_test.drop(columns=['CHAS'])
print(X2_train.shape, y_train.shape)
print(X2_test.shape, y_test.shape)


# #### Creating X1S with all 13 features scaled using StandardScaler

# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


std_scaler = StandardScaler()
X1S = std_scaler.fit_transform(df_boston_raw_data.iloc[:, :-1].values)


# In[22]:


X1S = pd.DataFrame(X1S, columns=boston_dataset_X_names)
X1S.head()


# In[23]:


X1S.describe().round(2)


# In[24]:


X1S_train, X1S_test = train_test_split(X1S, random_state=0)
print(X1S_train.shape, y_train.shape)
print(X1S_test.shape, y_test.shape)


# #### Creating X1N with all 13 features scaled using Normalizer

# In[25]:


from sklearn.preprocessing import Normalizer


# In[26]:


normalizer = Normalizer()
X1N = normalizer.fit_transform(df_boston_raw_data.iloc[:, :-1].values)


# In[27]:


X1N = pd.DataFrame(X1N, columns=boston_dataset_X_names)
X1N.head()


# In[28]:


X1N.describe().round(2)


# In[29]:


X1N_train, X1N_test = train_test_split(X1N, random_state=0)
print(X1N_train.shape, y_train.shape)
print(X1N_test.shape, y_test.shape)


# #### Creating X2S with 12 features scaled using StandardScaler

# In[30]:


X2S_train = X1S_train.drop(columns=['CHAS'])
X2S_test = X1S_test.drop(columns=['CHAS'])


# #### Creating X2N with 12 features scaled using Normalizer

# In[31]:


X2N_train = X1N_train.drop(columns=['CHAS'])
X2N_test = X1N_test.drop(columns=['CHAS'])


# In[32]:


model_attrib = {
    'model_names': [],
    'model_feature_counts': [],
    'model_feature_names': [],
    'model_r2_scores': [],
    'model_mae_scores': []
}


# #### Model 1: LR with X1

# In[33]:


lr_model_1 = LinearRegression()
lr_model_1.fit(X1_train, y_train)
lr_model_1_y_hat = lr_model_1.predict(X1_test)
model_attrib['model_names'].append('lr_model_1')
model_attrib['model_feature_counts'].append(X1_train.shape[1])
model_attrib['model_feature_names'].append(list(X1_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, lr_model_1_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, lr_model_1_y_hat))


# In[34]:


df_actuals_vs_predicted = pd.DataFrame({'actual': y_test, 'predicted': lr_model_1_y_hat})
df_actuals_vs_predicted


# #### Model 2: RFR with X1

# In[35]:


rf_model_1 = RandomForestRegressor(random_state=0)
rf_model_1.fit(X1_train, y_train)
rf_model_1_y_hat = rf_model_1.predict(X1_test)
model_attrib['model_names'].append('rf_model_1')
model_attrib['model_feature_counts'].append(X1_train.shape[1])
model_attrib['model_feature_names'].append(list(X1_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, rf_model_1_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, rf_model_1_y_hat))


# #### Model 3: LR with X2

# In[36]:


lr_model_2 = LinearRegression()
lr_model_2.fit(X2_train, y_train)
lr_model_2_y_hat = lr_model_2.predict(X2_test)
model_attrib['model_names'].append('lr_model_2')
model_attrib['model_feature_counts'].append(X2_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, lr_model_2_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, lr_model_2_y_hat))


# #### Model 4: RFR with X2

# In[37]:


rf_model_2 = RandomForestRegressor(random_state=0)
rf_model_2.fit(X2_train, y_train)
rf_model_2_y_hat = rf_model_2.predict(X2_test)
model_attrib['model_names'].append('rf_model_2')
model_attrib['model_feature_counts'].append(X2_train.shape[1])
model_attrib['model_feature_names'].append(list(X2_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, rf_model_2_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, rf_model_2_y_hat))


# #### Model 5: LR with X1S

# In[38]:


lr_model_3 = LinearRegression()
lr_model_3.fit(X1S_train, y_train)
lr_model_3_y_hat = lr_model_3.predict(X1S_test)
model_attrib['model_names'].append('lr_model_3')
model_attrib['model_feature_counts'].append(X1S_train.shape[1])
model_attrib['model_feature_names'].append(list(X1S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, lr_model_3_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, lr_model_3_y_hat))


# #### Model 6: RFR with X1S

# In[39]:


rf_model_3 = RandomForestRegressor(random_state=0)
rf_model_3.fit(X1S_train, y_train)
rf_model_3_y_hat = rf_model_3.predict(X1S_test)
model_attrib['model_names'].append('rf_model_3')
model_attrib['model_feature_counts'].append(X1S_train.shape[1])
model_attrib['model_feature_names'].append(list(X1S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, rf_model_3_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, rf_model_3_y_hat))


# In[40]:


show_model_eval_table(model_attrib)


# #### Model 7: LR with X2S

# In[41]:


lr_model_4 = LinearRegression()
lr_model_4.fit(X2S_train, y_train)
lr_model_4_y_hat = lr_model_4.predict(X2S_test)
model_attrib['model_names'].append('lr_model_4')
model_attrib['model_feature_counts'].append(X2S_train.shape[1])
model_attrib['model_feature_names'].append(list(X2S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, lr_model_4_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, lr_model_4_y_hat))


# #### Model 8: RFR with X2S

# In[42]:


rf_model_4 = RandomForestRegressor(random_state=0)
rf_model_4.fit(X2S_train, y_train)
rf_model_4_y_hat = rf_model_4.predict(X2S_test)
model_attrib['model_names'].append('rf_model_4')
model_attrib['model_feature_counts'].append(X2S_train.shape[1])
model_attrib['model_feature_names'].append(list(X2S_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, rf_model_4_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, rf_model_4_y_hat))


# #### Model 9: LR with X1N

# In[43]:


lr_model_5 = LinearRegression()
lr_model_5.fit(X1N_train, y_train)
lr_model_5_y_hat = lr_model_5.predict(X1N_test)
model_attrib['model_names'].append('lr_model_5')
model_attrib['model_feature_counts'].append(X1N_train.shape[1])
model_attrib['model_feature_names'].append(list(X1N_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, lr_model_5_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, lr_model_5_y_hat))


# #### Model 10: RFR with X1N

# In[44]:


rf_model_5 = RandomForestRegressor(random_state=0)
rf_model_5.fit(X1N_train, y_train)
rf_model_5_y_hat = rf_model_5.predict(X1N_test)
model_attrib['model_names'].append('rf_model_5')
model_attrib['model_feature_counts'].append(X1N_train.shape[1])
model_attrib['model_feature_names'].append(list(X1N_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, rf_model_5_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, rf_model_5_y_hat))


# #### Model 11: LR with X2N

# In[45]:


lr_model_6 = LinearRegression()
lr_model_6.fit(X2N_train, y_train)
lr_model_6_y_hat = lr_model_6.predict(X2N_test)
model_attrib['model_names'].append('lr_model_6')
model_attrib['model_feature_counts'].append(X2N_train.shape[1])
model_attrib['model_feature_names'].append(list(X2N_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, lr_model_6_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, lr_model_6_y_hat))


# #### Model 12: RFR with X2N

# In[46]:


rf_model_6 = RandomForestRegressor(random_state=0)
rf_model_6.fit(X2N_train, y_train)
rf_model_6_y_hat = rf_model_6.predict(X2N_test)
model_attrib['model_names'].append('rf_model_6')
model_attrib['model_feature_counts'].append(X2N_train.shape[1])
model_attrib['model_feature_names'].append(list(X2N_train.columns))
model_attrib['model_r2_scores'].append(r2_score(y_test, rf_model_6_y_hat))
model_attrib['model_mae_scores'].append(mean_absolute_error(y_test, rf_model_6_y_hat))


# In[47]:


show_model_eval_table(model_attrib)


# #### Note that we scale only X, not Y. If you forgot the reason for this, watch the video again!
# 
# #### And note that because of the above, using MAE to compare models that were created using differently scaled and unscaled data is also perfectly fine. Of course, R2 score is also fine. In fact, they should be used together to evaluate models.
