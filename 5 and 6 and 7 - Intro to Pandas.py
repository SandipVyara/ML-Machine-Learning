#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ## Problem 1 Statement
# 
# Given a series of revenues and expenses, analyze the profits.

# In[2]:


daily_revenues = [20, 22, 34, 31, 30, 22, 40, 41]
daily_expenses = [10, 12, 14, 21, 40, 2, 10, 14]


# ### Creating a Series

# In[3]:


sr_daily_revenues = pd.Series(daily_revenues)
sr_daily_expenses = pd.Series(daily_expenses)


# In[4]:


sr_daily_revenues


# ### Creating a Dataframe

# In[5]:


df_daily_revenues_and_expenses = pd.DataFrame({'daily_revenue': sr_daily_revenues, 
                                               'daily_expense': sr_daily_expenses})
df_daily_revenues_and_expenses


# ### Basic Data Wrangling: Calculate daily profits

# In[6]:


df_daily_revenues_and_expenses['daily_profit'] = df_daily_revenues_and_expenses.daily_revenue - df_daily_revenues_and_expenses.daily_expense


# ### Basic EDA using a few important properties and methods 

# In[7]:


df_daily_revenues_and_expenses


# In[8]:


df_daily_revenues_and_expenses.shape


# In[9]:


df_daily_revenues_and_expenses.info()


# In[10]:


df_daily_revenues_and_expenses.head()


# In[11]:


df_daily_revenues_and_expenses.tail()


# In[12]:


df_daily_revenues_and_expenses.nunique()


# In[13]:


df_daily_revenues_and_expenses.describe()


# In[14]:


df_daily_revenues_and_expenses.daily_profit.value_counts()


# In[15]:


df_daily_revenues_and_expenses.daily_profit.hist()


# ## Problem 2 Statement
# 
# Explore and understand the data from the last 5 batches of students.

# ### Read the data from given csv file

# In[16]:


file_path = '../data/demo1_historic.csv'

df_student_historic_data = pd.read_csv(file_path)


# ### EDA

# In[17]:


df_student_historic_data.head()


# In[18]:


df_student_historic_data.info()


# In[19]:


df_student_historic_data.nunique()


# In[20]:


df_student_historic_data.describe()


# In[21]:


df_student_historic_data.hist(figsize=(16, 12))


# In[22]:


df_student_historic_data.corr() # Pearson correlation coefficients


# ### Slicing dataframes using absolute numeric index

# In[23]:


df_slice_1 = df_student_historic_data.iloc[1:3, 0:4]
df_slice_1


# In[24]:


type(df_student_historic_data)


# In[25]:


type(df_slice_1)


# In[26]:


df_slice_1.iloc[0, 1] += 10
df_slice_1


# In[27]:


df_student_historic_data.iloc[1:3, 0:4]


# ### Slicing dataframes using names

# In[28]:


df_slice_1.loc[1, 'Age'] -= 10


# In[29]:


df_slice_1


# In[30]:


df_slice_2 = df_student_historic_data.loc[1:3, ['ID', 'Age', 'Is_Married', 'Has_Children']]
df_slice_2


# In[31]:


df_slice_3 = df_student_historic_data.loc[1:3, 'ID':'Has_Children']
df_slice_3


# ### Conditional slicing using loc

# In[32]:


df_slice_4 = df_student_historic_data.loc[df_student_historic_data.Is_Married == 0]
df_slice_4


# In[33]:


df_slice_5 = df_student_historic_data.loc[(df_student_historic_data.Is_Married == 0) & (df_student_historic_data.Drinks_Tea == 0)]
df_slice_5


# ### Conditional slicing using the query method

# In[34]:


df_slice_6 = df_student_historic_data.query('Is_Married == 0')
df_slice_6


# In[35]:


df_slice_7 = df_student_historic_data.query('(Is_Married == 0) and (Drinks_Tea == 0)')
df_slice_7


# ### A few more attributes

# In[36]:


list(df_student_historic_data.columns)


# In[37]:


print(list(df_student_historic_data.columns))


# In[38]:


df_student_historic_data.index


# In[39]:


print(list(df_student_historic_data.index))


# In[40]:


df_student_historic_data.values


# In[41]:


df_student_historic_data.T


# In[42]:


df_student_historic_data.T.info()


# In[43]:


df_student_historic_data.info()


# ### Creating a (deep) copy of a dataframe 

# In[44]:


df_student_historic_data_copy = df_student_historic_data.copy()


# ### Adding a column; apply() method to process data

# In[45]:


df_student_historic_data_copy['Age_in_Days'] = df_student_historic_data_copy.Age.apply(lambda x: int(x * 365.25))
df_student_historic_data_copy.head()


# ### Converting a column's data type using astype()

# In[46]:


df_student_historic_data_copy.Age = df_student_historic_data_copy.Age.astype(float)
df_student_historic_data_copy.head()


# ### A few more methods

# In[47]:


df_student_historic_data_copy.Exam_Result.value_counts()


# In[48]:


df_student_historic_data_copy.Exam_Result.replace(1, 0)


# In[49]:


df_student_historic_data_copy.head()


# In[50]:


df_student_historic_data_copy.Exam_Result.replace(1, 0, inplace=True)


# In[51]:


df_student_historic_data_copy.head()


# In[52]:


df_student_historic_data_copy.Exam_Result = df_student_historic_data_copy.Exam_Result.replace(0, 1)


# In[53]:


df_student_historic_data_copy.Exam_Result.value_counts()


# In[54]:


df_student_historic_data_copy_2 = df_student_historic_data.copy()


# In[55]:


df_student_historic_data_copy_2.Exam_Result.value_counts()


# ### TODO for YOU: replace all 1's to 0's and all 0's to 1's in df_student_historic_data_copy_2

# ### Adding a row to a dataframe

# In[56]:


df_student_historic_data_copy = df_student_historic_data.copy()
df_student_historic_data_copy.head(1)


# In[57]:


df_student_historic_data_copy.loc[1000] = ('S1', 41, 1, 1, 3, 23, 0, 1, 41, 1)


# In[58]:


df_student_historic_data_copy.tail()


# ### Finding and removing duplicate rows

# In[59]:


df_student_historic_data_copy.duplicated()


# In[60]:


df_student_historic_data_copy.duplicated().sum()


# In[61]:


df_student_historic_data_copy.loc[[0, 1000], :]


# In[62]:


df_student_historic_data_copy.drop_duplicates()


# In[63]:


df_student_historic_data_copy


# In[64]:


df_student_historic_data_copy.drop_duplicates(inplace=True)


# In[65]:


df_student_historic_data_copy.loc[1000] = ('S1', 41, 1, 1, 3, 23, 0, 1, 41, 1)


# In[66]:


df_student_historic_data_copy.drop_duplicates(inplace=True, keep='last')


# In[67]:


df_student_historic_data_copy


# In[68]:


# df_student_historic_data_copy.loc[0, :]

# will result in "KeyError: 0"


# In[69]:


df_student_historic_data_copy.iloc[0, :]


# In[70]:


df_student_historic_data_copy.reset_index()


# In[71]:


df_student_historic_data_copy.reset_index(inplace=True)
df_student_historic_data_copy


# In[72]:


df_student_historic_data_copy.reset_index(inplace=True, drop=True)
df_student_historic_data_copy


# ### Dataframe Grouping

# In[73]:


df_student_historic_data_copy_gb_age = df_student_historic_data_copy.groupby('Age')


# In[74]:


type(df_student_historic_data_copy)


# In[75]:


type(df_student_historic_data_copy_gb_age)


# In[76]:


df_student_historic_data_copy_gb_age.ngroups


# In[77]:


df_student_historic_data_copy_gb_age.groups


# In[78]:


for age_group, age_data in df_student_historic_data_copy_gb_age.groups.items():
    print('{} students are of age {}'.format(len(age_data), age_group))


# In[79]:


df_student_historic_data_copy.Age.value_counts()


# In[80]:


df_student_historic_data_copy_gb_age_g50 = df_student_historic_data_copy_gb_age.get_group(50)


# In[81]:


df_student_historic_data_copy_gb_age_g50


# ## Imputation (handling missing values)

# In[82]:


df_raw_data = pd.read_csv('../data/sample_dataset_1.csv', header=None)


# In[83]:


df_raw_data.shape


# In[84]:


df_raw_data


# In[85]:


df_raw_data.columns = ['item', 'user', 'rating']


# In[86]:


df_raw_data_2 = pd.read_csv('../data/sample_dataset_2.csv', header=None, names=['item', 'user', 'rating'])


# In[87]:


df_raw_data_2.shape


# In[88]:


df_raw_data_2


# In[89]:


df_raw_data_2.info()


# In[90]:


df_raw_data.info()


# In[91]:


df_raw_data_2.isnull()


# In[92]:


df_raw_data_2.isnull().sum()


# In[93]:


df_raw_data_2.isnull().sum().sum()


# In[94]:


# df_raw_data_2.user = df_raw_data_2.user.astype(int)

# will result in "ValueError: Cannot convert non-finite values (NA or inf) to integer"


# ### Imputation Technique 1: using fillna() with 'ffill'

# In[95]:


df_raw_data_2.item.fillna(method='ffill', inplace=True)
df_raw_data_2


# ### Imputation Technique 2: using fillna() with 'bfill'

# In[96]:


df_raw_data_2.user.fillna(method='bfill', inplace=True)
df_raw_data_2


# ### Imputation Technique 3: using fillna() with a fixed value

# In[97]:


df_raw_data_2.rating.fillna(value=df_raw_data_2.rating.mean(), inplace=True)
df_raw_data_2


# In[98]:


df_raw_data_2.user = df_raw_data_2.user.astype(int) # works now (after handling missing values)!


# In[99]:


df_raw_data_2.info()


# #### Imputation Technique 4: using replace() for non-NaN missing values

# In[100]:


df_po_raw_data = pd.read_csv('../data/post-operative.dat', header=None, skiprows=13, names=['L_CORE', 'L_SURF', 'L_O2', 'L_BP', 'SURF_STBL', 'CORE_STBL', 'BP_STBL', 'COMFORT', 'Decision'])


# In[101]:


df_po_raw_data.head()


# In[102]:


df_po_raw_data.info()


# In[103]:


df_po_raw_data.isnull().sum().sum()


# #### No missing values reported above, so employ other methods

# In[104]:


for i, c in enumerate(df_po_raw_data.columns):
    print('Variable: {}\n\tname: {}\n\ttype: {}'.format(i, c, type(df_po_raw_data.iloc[0, i])))
    print('\tUnique Values: {}'.format(df_po_raw_data[c].unique()))


# In[105]:


df_po_raw_data.COMFORT.value_counts()


# In[106]:


df_po_raw_data.query('COMFORT == "?"').shape[0]


# In[107]:


df_po_raw_data.query('COMFORT == "?"')


# #### Imputation for the COMFORT variable
# 
# <pre>
# Options available for handling missing data for this scenario:
#     
#     a) Replace with mean of values
#     b) Replace with median of values
#     c) Replace with mode of values
#     d) Replace with the previous of next values
#     e) Replace with random values
#     f) Drop rows with missing values
#     
# Lets assume (for now) that 'DECISION' has good correlation with 'COMFORT'. We have only 90 observations, of which 3 are missing (3.33%).
# 
# Option f) should be considered only when there are no other good choices.
# Option e) can be a good choice when a careful strategy is applied.
# Option d) seems to be a fair choice since it would not break anything that we know of.
# Option c) seems to be a good choice since it would not break anything and is more logical. 
# Options b) and a) can yield fractional numbers, so may not be a good choice.
# </pre>

# In[108]:


df_po_data = df_po_raw_data.copy()


# In[109]:


df_po_data.COMFORT.replace('?', df_po_data.COMFORT.mode()[0], inplace=True)


# In[110]:


df_po_data.COMFORT.value_counts()


# In[111]:


df_po_data.query('COMFORT == "?"')

