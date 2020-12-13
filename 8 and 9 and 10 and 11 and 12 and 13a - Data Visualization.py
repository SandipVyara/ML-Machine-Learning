#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_items_ratings = pd.read_csv('../data/sample_dataset_1.csv', header=None, names=['item', 'user', 'rating'])
df_items_ratings


# ## Task 1: Plot the average rating for each item

# ### Find unique items and their frequencies

# In[3]:


df_items_ratings.item.nunique()


# In[4]:


df_items_ratings.item.value_counts()


# ### Use groupby() to create groups of items

# In[5]:


df_items_ratings_gb_item = df_items_ratings.groupby('item')


# ### Calculate the average rating for each unique item (group) by iterating through each group

# In[6]:


average_ratings = []
for group_name, group_data in df_items_ratings_gb_item.groups.items():
    average_ratings.append((group_name, df_items_ratings_gb_item.get_group(group_name).rating.mean()))
average_ratings


# In[7]:


df_average_ratings = pd.DataFrame(average_ratings, columns=['item', 'avg_rating'])
df_average_ratings


# ### First, let's try a line plot, which is the default graph type in pandas.plot()

# In[8]:


df_average_ratings.plot()


# #### Many things are left desired in the plot above, such as proper label. But the most important thing to note is that the type of plot is not suitable for the case at hand.
# 
# #### Line charts are great to show change over time.

# ### Try a pie chart, maybe?

# In[9]:


df_average_ratings.plot(kind='pie', y='avg_rating')


# #### As we can see, pie chart is also not suitable for this case.
# 
# #### Pie charts are great to show proportions.

# ### Plot a barchart, which is the most appropriate type for this particular case.

# In[10]:


df_average_ratings.plot(kind='barh')


# ### Set the item name as index, so that the barchart would get the labels automatically

# In[11]:


df_average_ratings.set_index('item', inplace=True)
df_average_ratings


# In[12]:


df_average_ratings.plot(kind='barh')


# In[13]:


df_average_ratings.plot(kind='barh', legend=False, title='Average Ratings')


# In[14]:


df_average_ratings.sort_values(by='avg_rating').plot(kind='barh', legend=False, title='Average Ratings')


# ## Task 2: plot the sales amounts for each month

# In[15]:


df_shampoo_sales_data = pd.read_csv('../data/shampoo_sales_data.csv')
df_shampoo_sales_data


# ### Line chart is the most appropriate for this case, try a default line chart first

# In[16]:


df_shampoo_sales_data.plot()


# ### Set proper labels

# In[17]:


df_shampoo_sales_data.set_index('Month', inplace=True)


# In[18]:


df_shampoo_sales_data.plot()


# ### Let's see what happens when you use a sub-optimal graph type for this particular case

# In[19]:


df_shampoo_sales_data.plot(kind='hist')


# #### As you can see this is not the best idea! As it does not sove the particular task of showing sales figures for each month. 
# 
# #### However, it can present the same data with a different note if done properly.

# In[20]:


df_shampoo_sales_data.loc[(df_shampoo_sales_data.Sales > 100) & (df_shampoo_sales_data.Sales < 180)]


# #### Fix the bins and the bar alignments, also add lables

# In[21]:


import numpy as np
import matplotlib.pyplot as plt


# In[22]:


np.histogram_bin_edges(df_shampoo_sales_data)


# In[23]:


def align_bin_edges(df, figure_size, plot_title, x_label):
    bin_edges = np.histogram_bin_edges(df)
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    plt.xticks(bin_edges)
    return ax, bin_edges


# In[24]:


ax1, be1 = align_bin_edges(df_shampoo_sales_data, (12, 9), '# of months per sales amount bucket', 'Amount')
df_shampoo_sales_data.plot(kind='hist', ax=ax1, bins=be1, legend=False)


# In[25]:


df_shampoo_sales_data.describe()


# ### Make a better line chart

# In[26]:


df_shampoo_sales_data.plot(figsize=(12, 9), grid=True, title='Sales per Month', legend=False, marker='^')
plt.xticks(rotation=90, ticks=np.arange(0, 36), labels=df_shampoo_sales_data.index.tolist())


# ## Task 3: EDA for 'car' dataset

# In[27]:


df_car_eval_data = pd.read_csv('../data/car.data', header=None, 
                               names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car_eval'])


# ### Text EDA

# In[28]:


df_car_eval_data.head()


# #### In a dataset, normally you have 2 types of variables:
# 
# <pre>
# X, also called Features, or Independent Variables, Input Variables, Source Variable
# Y, also called Label, or Dependent Variable, Output Variable, Target Variable
# </pre>

# In[29]:


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


# In[30]:


df_car_eval_data_info, df_car_eval_data_mem = get_df_info(df_car_eval_data, include_unique_values=True)
print('{} has {} row and {} cols, uses approx. {:.2f} MB'.format('df_car_eval_data', df_car_eval_data.shape[0],
                                                                 df_car_eval_data.shape[1], df_car_eval_data_mem))
df_car_eval_data_info


# In[31]:


df_car_eval_data.info()


# #### Data Wrangling for further EDA

# In[32]:


df_car_eval_data_cleaned = df_car_eval_data.copy()


# In[33]:


df_car_eval_data_cleaned = df_car_eval_data_cleaned.astype('category')


# In[34]:


df_car_eval_data_cleaned.head()


# In[35]:


df_car_eval_data_cleaned_info, df_car_eval_data_cleaned_mem = get_df_info(df_car_eval_data_cleaned, include_unique_values=True)
print('{} has {} row and {} cols, uses approx. {:.2f} MB'.format('df_car_eval_data_cleaned', df_car_eval_data_cleaned.shape[0],
                                                                 df_car_eval_data_cleaned.shape[1], df_car_eval_data_cleaned_mem))
df_car_eval_data_cleaned_info


# In[36]:


for c in df_car_eval_data_cleaned.columns:
    df_car_eval_data_cleaned[c] = df_car_eval_data_cleaned[c].cat.codes


# In[37]:


df_car_eval_data_cleaned.head()


# In[38]:


df_car_eval_data_cleaned_info, df_car_eval_data_cleaned_mem = get_df_info(df_car_eval_data_cleaned, include_unique_values=True)
print('{} has {} row and {} cols, uses approx. {:.2f} MB'.format('df_car_eval_data_cleaned', df_car_eval_data_cleaned.shape[0],
                                                                 df_car_eval_data_cleaned.shape[1], df_car_eval_data_cleaned_mem))
df_car_eval_data_cleaned_info


# ### Correlation analysis

# In[39]:


df_car_eval_data_cleaned_corr = df_car_eval_data_cleaned.corr()
df_car_eval_data_cleaned_corr


# #### The rather unique property of this dataset as you notices is 0 corr between all x variables. This is quite rare in real-world datsets!

# #### Plot the corr table

# In[40]:


import matplotlib.pyplot as plt


# In[41]:


plt.imshow(df_car_eval_data_cleaned_corr)


# #### Improve the above heatmap

# In[42]:


plt.imshow(df_car_eval_data_cleaned_corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(df_car_eval_data_cleaned_corr)), df_car_eval_data_cleaned_corr.columns, rotation=90)
plt.yticks(range(len(df_car_eval_data_cleaned_corr)), df_car_eval_data_cleaned_corr.index)


# #### A better and easier heatmap using Seaborn

# In[43]:


import seaborn as sb


# In[44]:


sb.heatmap(df_car_eval_data_cleaned_corr)


# In[45]:


sb.heatmap(df_car_eval_data_cleaned_corr, cmap='coolwarm', annot=True, vmin=-1, vmax=1)


# #### Some aspects of the corrections are difficult to interpret and also will probably cause errors in further EDA, as a result of the automatic mapping from original string values to numeric values using category codes. 
# 
# #### Hence, we shall create custom mapping and redo corr analysis for correct understand and reflection of correlations.

# In[46]:


df_car_eval_data_numeric = df_car_eval_data.copy()


# In[47]:


df_car_eval_data_numeric_info, df_car_eval_data_numeric_mem = get_df_info(df_car_eval_data_numeric, include_unique_values=True)
print('{} has {} row and {} cols, uses approx. {:.2f} MB'.format('df_car_eval_data_numeric', df_car_eval_data_numeric.shape[0],
                                                                 df_car_eval_data_numeric.shape[1], df_car_eval_data_numeric_mem))
df_car_eval_data_numeric_info


# In[48]:


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


# In[49]:


for c in df_car_eval_data_numeric.columns:
    df_car_eval_data_numeric['x_' + c] = df_car_eval_data_numeric[c].apply(lambda x: get_cat_codes(x, c))


# In[50]:


df_car_eval_data_numeric.head()


# In[51]:


df_car_eval_data_numeric_info, df_car_eval_data_numeric_mem = get_df_info(df_car_eval_data_numeric, include_unique_values=True)
print('{} has {} row and {} cols, uses approx. {:.2f} MB'.format('df_car_eval_data_numeric', df_car_eval_data_numeric.shape[0],
                                                                 df_car_eval_data_numeric.shape[1], df_car_eval_data_numeric_mem))
df_car_eval_data_numeric_info


# In[52]:


df_car_eval_data_numeric = df_car_eval_data_numeric.iloc[:, 7:]


# In[53]:


df_car_eval_data_numeric.columns = df_car_eval_data.columns


# In[54]:


df_car_eval_data_numeric


# In[55]:


sb.heatmap(df_car_eval_data_numeric.corr(), cmap='coolwarm', annot=True, vmin=-1, vmax=1)


# #### Now you can see that these correlations between each x and y make sense!

# #### TODO FOR YOU: Do you think there is just 'no correlation' between the features, or more than just 'no correlation'? 
# 
# If yes, what would you call it - 'no association' or maybe 'no causation'? 
# If no, why?

# ### Undertanding relationship between variables using Scatterplots

# In[56]:


sb.scatterplot(x='safety', y='car_eval', data=df_car_eval_data_numeric)
plt.xticks(ticks=np.arange(0, 3), labels=['low', 'med', 'high'])
plt.yticks(ticks=np.arange(0, 4), labels=['unacc', 'acc', 'good', 'vgood'])


# In[57]:


sb.scatterplot(x='doors', y='car_eval', data=df_car_eval_data_numeric)
plt.xticks(ticks=np.arange(0, 4), labels=['2', '3', '4', '5+'])
plt.yticks(ticks=np.arange(0, 4), labels=['unacc', 'acc', 'good', 'vgood'])


# ### At a glance using a pairplot - a fusion of multiple scatterplots and histograms

# In[58]:


sb.pairplot(df_car_eval_data_numeric)


# In[59]:


sb.pairplot(df_car_eval_data_numeric, vars=['lug_boot', 'safety', 'car_eval'])


# ### Single scatterplot with linear regression modelling 

# In[60]:


sb.lmplot(x='safety', y='car_eval', data=df_car_eval_data_numeric)


# In[61]:


sb.lmplot(x='doors', y='car_eval', data=df_car_eval_data_numeric)


# ### Multiple scatterplots with faceting

# In[62]:


sb.relplot(x='safety', y='doors', col='car_eval', col_wrap=2, data=df_car_eval_data_numeric)


# ## Task 4: EDA for 'post-operative' dataset

# In[63]:


df_po_raw_data = pd.read_csv('../data/post-operative.dat', header=None, names=['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'Decision'], skiprows=13)


# ### Text-based EDA

# In[64]:


df_po_raw_data.head()


# In[65]:


df_po_raw_data_info, df_po_raw_data_mem = get_df_info(df_po_raw_data, include_unique_values=True)
print('{} has {} rows and {} cols, uses approx. {:.2f} MB'.format('df_po_raw_data_pp', df_po_raw_data.shape[0], 
                                                                  df_po_raw_data.shape[1], df_po_raw_data_mem))
df_po_raw_data_info


# ### Data Pre-Processing

# In[66]:


df_po_raw_data_pp = df_po_raw_data.copy()
df_po_raw_data_pp.COMFORT.replace('?', df_po_raw_data_pp.COMFORT.mode()[0], inplace=True)


# In[67]:


def get_cat_codes_po(str_val):
    if str_val in ['low', 'unstable', '05', 'I']:
        return 0
    if str_val in ['high', 'good', 'mod-stable', '07', '10', 'A']:
        return 1
    if str_val in ['mid', 'excellent', 'stable', '15', 'S']:
        return 2      


# In[68]:


for c in df_po_raw_data_pp.columns:
    df_po_raw_data_pp['x_' + c] = df_po_raw_data_pp[c].apply(lambda x: get_cat_codes_po(x))


# In[69]:


df_po_raw_data_pp_info, df_po_raw_data_pp_mem = get_df_info(df_po_raw_data_pp, include_unique_values=True)
print('{} has {} rows and {} cols, uses approx. {:.2f} MB'.format('df_po_raw_data_pp', df_po_raw_data_pp.shape[0], 
                                                                  df_po_raw_data_pp.shape[1], df_po_raw_data_pp_mem))
df_po_raw_data_pp_info


# In[70]:


df_po_raw_data_pp.drop(df_po_raw_data.columns, axis=1, inplace=True)
df_po_raw_data_pp.columns = df_po_raw_data.columns


# In[71]:


df_po_raw_data_pp.head()


# ### Visual EDA

# In[72]:


sb.pairplot(df_po_raw_data_pp)


# In[73]:


sb.lmplot(x='BP-STBL', y='CORE-STBL', data=df_po_raw_data_pp)


# ## Task 5: EDA for 'diamonds' dataset

# In[74]:


df_diamonds_raw_data = pd.read_csv('../data/diamonds.csv')


# ### Text EDA

# In[75]:


df_diamonds_raw_data.head()


# In[76]:


df_diamonds_raw_data_info, df_diamonds_raw_data_mem = get_df_info(df_diamonds_raw_data, include_unique_values=True)
print('{} has {} rows and {} cols, uses approx. {:.2f} MB'.format('df_diamonds_raw_data', df_diamonds_raw_data.shape[0], 
                                                                  df_diamonds_raw_data.shape[1], df_diamonds_raw_data_mem))
df_diamonds_raw_data_info


# In[77]:


df_diamonds_raw_data.describe()


# ### Viz EDA

# In[78]:


sb.distplot(df_diamonds_raw_data.carat)

# (right-skewed) asymmetric claw distribution, 
#  but can be approximated to a log-normal distribution


# In[79]:


sb.distplot(df_diamonds_raw_data.table)

# claw distribution, can be approximated to a normal distribution


# In[80]:


sb.distplot(df_diamonds_raw_data.price)

#  this does not follow the standard-normal distribution, as it is heavily skewed to the right 
#  - a log-normal distribution


# #### Pairplot with distribution plot: 
#      a) understanding the distribution of each variable 
#      b) understanding the replation of each variable to another 
#         - direction: positive or negative or neutral 
#         - strength: strong or weak 
#         - pattern: linear or non-linear

# In[81]:


sb.pairplot(data=df_diamonds_raw_data, diag_kind='kde')


# ### Illlustration of the Central Limit Theorum (CLT)
# 
# CLT: Regardless of the distribution of a certain sample of data, with a large enough sample the distribution approaches the normal distribution (for many cases).

# #### Create samples of a feature using the sample() method
# 
# <pre>
# syntax: .sample(n, random_state): 
#   n is the number of observations you want in the sample 
#   random_state is the seed for random sampling, useful for reproducing results
# </pre>

# In[82]:


depth_sample_list = [df_diamonds_raw_data.depth.sample(n=10, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=25, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=50, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=100, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=200, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=300, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=400, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=500, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=600, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=700, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=800, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=900, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=1000, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=2500, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=5000, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=10000, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=20000, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=30000, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.sample(n=40000, random_state=0).to_numpy(),
                     df_diamonds_raw_data.depth.to_numpy()]


# In[83]:


sb.distplot(depth_sample_list[0]) # right-skewed distribution that somewhat resembles a normal distribution


# In[84]:


sb.distplot(depth_sample_list[10]) # more like a normal distribution now


# In[85]:


sb.distplot(depth_sample_list[-1]) # much more like a normal distribution


# #### You can see that the data distribution progresses towards the 'normal' distribution as more observations are available in the sample

# In[86]:


import matplotlib.animation as mpani
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[87]:


def ani_func(current):
    plt.cla()
    plt.hist(depth_sample_list[current], bins=15, histtype='stepfilled', range=(55, 69))
    plt.gca().set_title('N = {}'.format(len(depth_sample_list[current])))
    if current == 19:
        ani_obj.event_source.stop()


# In[88]:


fig = plt.figure()
ani_obj = mpani.FuncAnimation(fig, ani_func, interval=2000)

