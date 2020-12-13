#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb


# In[2]:


def get_df_info(df, include_unique_values=False):
    col_name_list = list(df.columns)
    col_type_list = [type(cell) for cell in df.iloc[0, :]]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_mem_usage_list = [df[col].memory_usage(deep=True) for col in col_name_list]
    total_memory_usage = sum(col_mem_usage_list) / 1048576
    if include_unique_values:
        col_unique_list = [df[col].unique() for col in col_name_list]
        df_info = pd.DataFrame({'column_name': col_name_list, 'type': col_type_list, 
                                'null_count': col_null_count_list, 'nunique': col_unique_count_list, 
                                'unique_values': col_unique_list})
    else:
        df_info = pd.DataFrame({'column_name': col_name_list, 'type': col_type_list, 
                        'null_count': col_null_count_list, 'nunique': col_unique_count_list})
    return df_info, total_memory_usage


# ## Task 6: EDA for 'pokemon' dataset

# In[3]:


df_pokemon_raw_data = pd.read_csv('../data/Pokemon.csv', encoding='latin1')


# ### Text EDA

# In[4]:


df_pokemon_raw_data.head()


# In[5]:


df_pokemon_raw_data_info, df_pokemon_raw_data_mem = get_df_info(df_pokemon_raw_data, include_unique_values=True)
print('{} has {} row and {} cols, uses approx. {:.2f} MB'.format('df_pokemon_raw_data', df_pokemon_raw_data.shape[0],
                                                                 df_pokemon_raw_data.shape[1], df_pokemon_raw_data_mem))
df_pokemon_raw_data_info


# ### Viz EDA

# In[6]:


sb.lmplot(x='Attack', y='Defense', data=df_pokemon_raw_data)


# In[7]:


sb.lmplot(x='Attack', y='Defense', data=df_pokemon_raw_data, fit_reg=False, hue='Legendary')


# In[8]:


sb.lmplot(x='Attack', y='Defense', data=df_pokemon_raw_data, fit_reg=False, hue='Stage')


# In[9]:


sb.relplot(x='HP', y='Speed', data=df_pokemon_raw_data, col='Type 1', col_wrap=5)


# ### Boxplot to check:
# 
# ### - min and max (whisker boundaries)
# ### - 1st, 2nd and 3rd quartiles (box boundaries)
# ### - outliers (isolated diamonds)

# In[10]:


sb.set(rc={'figure.figsize': (12, 9)})
sb.boxplot(data=df_pokemon_raw_data)


# In[11]:


df_pokemon_raw_data.describe()


# In[12]:


sb.boxplot(data=df_pokemon_raw_data.Stage)


# ### Stripplot shows data points with distribution and density, useful for comparing variables

# In[13]:


sb.stripplot(data=df_pokemon_raw_data)


# In[14]:


df_pokemon_raw_data.Total.value_counts()[:10]

