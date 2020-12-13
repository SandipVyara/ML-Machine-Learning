#!/usr/bin/env python
# coding: utf-8

# <img src="19 and 20 - Naive Bayes.jpg" alt="Drawing" style="width: 1200px;"/>

# <img src="20 - Logistic Regression.jpg" alt="Drawing" style="width: 1200px;"/>

# <img src="21 - Decision Tree.jpg" alt="Drawing" style="width: 1200px;"/>

# In[1]:


import pandas as pd


# In[2]:


df_student_historic_raw_data = pd.read_csv('../data/demo1_historic.csv')


# ## 1 - EDA

# In[3]:


def get_df_info(df, include_unique_values=False):
    col_name_list = list(df.columns)
    col_type_list = [type(cell) for cell in df.iloc[0, :]]
    col_null_count_list = [df[col].isnull().sum() for col in col_name_list]
    col_unique_count_list = [df[col].nunique() for col in col_name_list]
    col_memory_usage_list = [df[col].memory_usage() for col in col_name_list]
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


df_student_historic_raw_data.head()


# In[5]:


import seaborn as sb


# In[6]:


sb.set(rc={'figure.figsize':(12,9)})
sb.heatmap(df_student_historic_raw_data.drop('ID', axis=1).corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True)


# ## 2 - Preprocessing (not required as all variables are ints and no missing data)

# ## 3 - Feature Selection

# <pre>
# x                    -> y
# features             -> labels
# input variable       -> output variable 
# independent variable -> dependent variable
# predictor variable   -> response variable
# source variable      -> target variable
# </pre>

# In[7]:


labels = df_student_historic_raw_data.loc[:, 'Exam_Result']
labels.shape


# In[8]:


labels.value_counts()


# ### Feature set 1: all features (except ID)

# In[9]:


df_feature_set_1 = df_student_historic_raw_data.drop(['ID', 'Exam_Result'], axis=1)
df_feature_set_1.shape


# ### Feature set 2: based on correlation analysis (removed 2 dependent features)

# In[10]:


df_feature_set_2 = df_student_historic_raw_data.drop(['ID', 'Exam_Result', 'Children_Count', 'Is_Married'], axis=1)
df_feature_set_2.shape


# ## 4 - Data Split

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X1_train, X1_test, y_train, y_test = train_test_split(df_feature_set_1, labels, random_state=0)
print(X1_train.shape, y_train.shape)
print(X1_test.shape, y_test.shape)


# In[13]:


X2_train, X2_test = train_test_split(df_feature_set_2, random_state=0)
print(X2_train.shape, y_train.shape)
print(X2_test.shape, y_test.shape)


# ## 5 - Create and evaluate models

# In[14]:


from sklearn.linear_model import LogisticRegression as LRC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score


# In[15]:


model_names = []
model_feature_counts = []
model_feature_names = []
model_acc_scores = []
model_rec_scores = []
model_bac_scores = []


# ### Model 1: LRC with X1

# In[16]:


lr_model_1 = LRC(random_state=0)
lr_model_1.fit(X1_train, y_train)


# In[17]:


lr_model_1_y_hat = lr_model_1.predict(X1_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(lr_model_1_y_hat, name='Predicted')
lr_model_1_cm = pd.crosstab(y_actual, y_predicted)
print(lr_model_1_cm, '\n')
lr_model_1_acc = accuracy_score(y_test, lr_model_1_y_hat)
lr_model_1_bac = balanced_accuracy_score(y_test, lr_model_1_y_hat)
lr_model_1_rec = recall_score(y_test, lr_model_1_y_hat)
print('ACC: {:>.2f}\tBAC: {:>.2f}\tREC: {:>.2f}'.format(lr_model_1_acc, lr_model_1_bac, lr_model_1_rec))
model_names.append('lr_model_1')
model_feature_counts.append(X1_train.shape[1])
model_feature_names.append(list(X1_train.columns))
model_acc_scores.append(lr_model_1_acc)
model_bac_scores.append(lr_model_1_bac)
model_rec_scores.append(lr_model_1_rec)


# <pre>
# A   P
# 0   0  -> TN -> Good
# 0   1  -> FP -> Bad
# 1   0  -> FN -> Bad
# 1   1  -> TP -> Good
# 
# ACC = (TP + TN) / (TP + TN + FP + FN) = (25 + 140) / 250 = 0.66 = 66%
# REC (True Positive Rate) = TP / (TP + FN) = 140 / (140 + 13) = 140 / 153 = 0.92 = 92%
# TNR = TN / (TN + FP) = 25 / (25 + 72) = 25 / 97 = 0.26 = 26%
# BAC = (TNR + TPR) / 2 = (0.92 + 0.26) / 2 = 0.59 = 59%
# </pre>

# ### Model 2: DTC with X1

# In[18]:


dt_model_1 = DTC(random_state=0)
dt_model_1.fit(X1_train, y_train)


# In[19]:


dt_model_1_y_hat = dt_model_1.predict(X1_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(dt_model_1_y_hat, name='Predicted')
dt_model_1_cm = pd.crosstab(y_actual, y_predicted)
print(dt_model_1_cm, '\n')
dt_model_1_acc = accuracy_score(y_test, dt_model_1_y_hat)
dt_model_1_bac = balanced_accuracy_score(y_test, dt_model_1_y_hat)
dt_model_1_rec = recall_score(y_test, dt_model_1_y_hat)
print('ACC: {:>.2f}\tBAC: {:>.2f}\tREC: {:>.2f}'.format(dt_model_1_acc, dt_model_1_bac, dt_model_1_rec))
model_names.append('dt_model_1')
model_feature_counts.append(X1_train.shape[1])
model_feature_names.append(list(X1_train.columns))
model_acc_scores.append(dt_model_1_acc)
model_bac_scores.append(dt_model_1_bac)
model_rec_scores.append(dt_model_1_rec)


# ### Model 3: LRC with X2

# In[20]:


lr_model_2 = LRC(random_state=0)
lr_model_2.fit(X2_train, y_train)
lr_model_2_y_hat = lr_model_2.predict(X2_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(lr_model_2_y_hat, name='Predicted')
lr_model_2_cm = pd.crosstab(y_actual, y_predicted)
print(lr_model_2_cm, '\n')
lr_model_2_acc = accuracy_score(y_test, lr_model_2_y_hat)
lr_model_2_bac = balanced_accuracy_score(y_test, lr_model_2_y_hat)
lr_model_2_rec = recall_score(y_test, lr_model_2_y_hat)
print('ACC: {:>.2f}\tBAC: {:>.2f}\tREC: {:>.2f}'.format(lr_model_2_acc, lr_model_2_bac, lr_model_2_rec))
model_names.append('lr_model_2')
model_feature_counts.append(X2_train.shape[1])
model_feature_names.append(list(X2_train.columns))
model_acc_scores.append(lr_model_2_acc)
model_bac_scores.append(lr_model_2_bac)
model_rec_scores.append(lr_model_2_rec)


# ### Model 4: DTC with X2

# In[21]:


dt_model_2 = DTC(random_state=0)
dt_model_2.fit(X2_train, y_train)
dt_model_2_y_hat = dt_model_2.predict(X2_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(dt_model_2_y_hat, name='Predicted')
dt_model_2_cm = pd.crosstab(y_actual, y_predicted)
print(dt_model_2_cm, '\n')
dt_model_2_acc = accuracy_score(y_test, dt_model_2_y_hat)
dt_model_2_bac = balanced_accuracy_score(y_test, dt_model_2_y_hat)
dt_model_2_rec = recall_score(y_test, dt_model_2_y_hat)
print('ACC: {:>.2f}\tBAC: {:>.2f}\tREC: {:>.2f}'.format(dt_model_2_acc, dt_model_2_bac, dt_model_2_rec))
model_names.append('dt_model_2')
model_feature_counts.append(X2_train.shape[1])
model_feature_names.append(list(X2_train.columns))
model_acc_scores.append(dt_model_2_acc)
model_bac_scores.append(dt_model_2_bac)
model_rec_scores.append(dt_model_2_rec)


# ### Model 5: RFC with X1

# In[22]:


from sklearn.ensemble import RandomForestClassifier as RFC


# In[23]:


rf_model_1 = RFC(random_state=0)
rf_model_1.fit(X1_train, y_train)
rf_model_1_y_hat = rf_model_1.predict(X1_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(rf_model_1_y_hat, name='Predicted')
rf_model_1_cm = pd.crosstab(y_actual, y_predicted)
print(rf_model_1_cm, '\n')
rf_model_1_acc = accuracy_score(y_test, rf_model_1_y_hat)
rf_model_1_bac = balanced_accuracy_score(y_test, rf_model_1_y_hat)
rf_model_1_rec = recall_score(y_test, rf_model_1_y_hat)
print('ACC: {:>.2f}\tBAC: {:>.2f}\tREC: {:>.2f}'.format(rf_model_1_acc, rf_model_1_bac, rf_model_1_rec))
model_names.append('rf_model_1')
model_feature_counts.append(X1_train.shape[1])
model_feature_names.append(list(X1_train.columns))
model_acc_scores.append(rf_model_1_acc)
model_bac_scores.append(rf_model_1_bac)
model_rec_scores.append(rf_model_1_rec)


# ### Model 6: RFC with X2

# In[24]:


rf_model_2 = RFC(random_state=0)
rf_model_2.fit(X2_train, y_train)
rf_model_2_y_hat = rf_model_2.predict(X2_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(rf_model_2_y_hat, name='Predicted')
rf_model_2_cm = pd.crosstab(y_actual, y_predicted)
print(rf_model_2_cm, '\n')
rf_model_2_acc = accuracy_score(y_test, rf_model_2_y_hat)
rf_model_2_bac = balanced_accuracy_score(y_test, rf_model_2_y_hat)
rf_model_2_rec = recall_score(y_test, rf_model_2_y_hat)
print('ACC: {:>.2f}\tBAC: {:>.2f}\tREC: {:>.2f}'.format(rf_model_2_acc, rf_model_2_bac, rf_model_2_rec))
model_names.append('rf_model_2')
model_feature_counts.append(X2_train.shape[1])
model_feature_names.append(list(X2_train.columns))
model_acc_scores.append(rf_model_2_acc)
model_bac_scores.append(rf_model_2_bac)
model_rec_scores.append(rf_model_2_rec)


# In[25]:


df_model_eval = pd.DataFrame({'model': model_names, 'feature_count': model_feature_counts, 
                              'feature_names': model_feature_names, 'acc': model_acc_scores, 
                              'bac': model_bac_scores, 'rec': model_rec_scores})
df_model_eval.round(2)


# #### Analyze the feature importances as reported by the RFC model we created above

# In[26]:


rf_model_1.feature_importances_


# In[27]:


df_fi_X1_rf1 = pd.DataFrame({'feature': list(X1_test.columns), 
                             'importance': list(rf_model_1.feature_importances_)})
df_fi_X1_rf1.sort_values(by='importance', inplace=True)
df_fi_X1_rf1.set_index('feature', drop=True, inplace=True)
df_fi_X1_rf1


# In[28]:


df_fi_X1_rf1.plot(kind='barh')


# In[29]:


df_fi_X2_rf2 = pd.DataFrame({'feature': list(X2_test.columns), 
                             'importance': list(rf_model_2.feature_importances_)})
df_fi_X2_rf2.sort_values(by='importance', inplace=True)
df_fi_X2_rf2.set_index('feature', drop=True, inplace=True)
df_fi_X2_rf2


# In[30]:


df_fi_X2_rf2.plot(kind='barh')


# ### Feature set 3: based on RFC.feature_importances_ (keeping only top 4, both models show the same top 4 features)

# In[31]:


df_feature_set_3 = df_student_historic_raw_data.loc[:, ['Average_Assignments_Score', 'Home_Distance', 'Age', 'Lectures_Missed']]
df_feature_set_3.shape


# In[32]:


X3_train, X3_test = train_test_split(df_feature_set_3, random_state=0)
print(X3_train.shape, y_train.shape)
print(X3_test.shape, y_test.shape)


# ### Model 7: LRC with X3

# In[33]:


lr_model_3 = LRC(random_state=0)
lr_model_3.fit(X3_train, y_train)
lr_model_3_y_hat = lr_model_3.predict(X3_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(lr_model_3_y_hat, name='Predicted')
lr_model_3_cm = pd.crosstab(y_actual, y_predicted)
print(lr_model_3_cm, '\n')
lr_model_3_acc = accuracy_score(y_test, lr_model_3_y_hat)
lr_model_3_bac = balanced_accuracy_score(y_test, lr_model_3_y_hat)
lr_model_3_rec = recall_score(y_test, lr_model_3_y_hat)
print('ACC: {:>.3f}\tBAC: {:>.3f}\tREC: {:>.3f}'.format(lr_model_3_acc, lr_model_3_bac, lr_model_3_rec))
model_names.append('lr_model_3')
model_feature_counts.append(X3_train.shape[1])
model_feature_names.append(list(X3_train.columns))
model_acc_scores.append(lr_model_3_acc)
model_bac_scores.append(lr_model_3_bac)
model_rec_scores.append(lr_model_3_rec)


# ### Model 8: DTC with X3

# In[34]:


dt_model_3 = DTC(random_state=0)
dt_model_3.fit(X3_train, y_train)
dt_model_3_y_hat = dt_model_3.predict(X3_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(dt_model_3_y_hat, name='Predicted')
dt_model_3_cm = pd.crosstab(y_actual, y_predicted)
print(dt_model_3_cm, '\n')
dt_model_3_acc = accuracy_score(y_test, dt_model_3_y_hat)
dt_model_3_bac = balanced_accuracy_score(y_test, dt_model_3_y_hat)
dt_model_3_rec = recall_score(y_test, dt_model_3_y_hat)
print('ACC: {:>.3f}\tBAC: {:>.3f}\tREC: {:>.3f}'.format(dt_model_3_acc, dt_model_3_bac, dt_model_3_rec))
model_names.append('dt_model_3')
model_feature_counts.append(X3_train.shape[1])
model_feature_names.append(list(X3_train.columns))
model_acc_scores.append(dt_model_3_acc)
model_bac_scores.append(dt_model_3_bac)
model_rec_scores.append(dt_model_3_rec)


# ### Model 9: RFC with X3

# In[35]:


rf_model_3 = RFC(random_state=0)
rf_model_3.fit(X3_train, y_train)
rf_model_3_y_hat = rf_model_3.predict(X3_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(rf_model_3_y_hat, name='Predicted')
rf_model_3_cm = pd.crosstab(y_actual, y_predicted)
print(rf_model_3_cm, '\n')
rf_model_3_acc = accuracy_score(y_test, rf_model_3_y_hat)
rf_model_3_bac = balanced_accuracy_score(y_test, rf_model_3_y_hat)
rf_model_3_rec = recall_score(y_test, rf_model_3_y_hat)
print('ACC: {:>.3f}\tBAC: {:>.3f}\tREC: {:>.3f}'.format(rf_model_3_acc, rf_model_3_bac, rf_model_3_rec))
model_names.append('rf_model_3')
model_feature_counts.append(X3_train.shape[1])
model_feature_names.append(list(X3_train.columns))
model_acc_scores.append(rf_model_3_acc)
model_bac_scores.append(rf_model_3_bac)
model_rec_scores.append(rf_model_3_rec)


# In[36]:


df_model_eval = pd.DataFrame({'model': model_names, 'feature_count': model_feature_counts, 
                              'feature_names': model_feature_names, 'acc': model_acc_scores, 
                              'bac': model_bac_scores, 'rec': model_rec_scores})
df_model_eval.round(2)


# #### Assume that for this project, the primary metric is BAC and the secondary metric is REC.
# #### Then you can evaluate the models using a weighted average, for example: (.60 * bac + .40 * rec) / 2
# #### The decision as to what weights to use would come from your understanding of the data and business requirements

# ### Feature set 4: based on analysis of both RFC.feature_importances_ and correlation matrix (keeping only top 3)

# In[37]:


df_feature_set_4 = df_student_historic_raw_data.loc[:, ['Average_Assignments_Score', 'Age', 'Lectures_Missed']]
df_feature_set_4.shape


# In[38]:


X4_train, X4_test = train_test_split(df_feature_set_4, random_state=0)
print(X4_train.shape, y_train.shape)
print(X4_test.shape, y_test.shape)


# ### Model 10: LRC with X4

# In[39]:


lr_model_4 = LRC(random_state=0)
lr_model_4.fit(X4_train, y_train)
lr_model_4_y_hat = lr_model_4.predict(X4_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(lr_model_4_y_hat, name='Predicted')
lr_model_4_cm = pd.crosstab(y_actual, y_predicted)
print(lr_model_4_cm, '\n')
lr_model_4_acc = accuracy_score(y_test, lr_model_4_y_hat)
lr_model_4_bac = balanced_accuracy_score(y_test, lr_model_4_y_hat)
lr_model_4_rec = recall_score(y_test, lr_model_4_y_hat)
print('ACC: {:>.4f}\tBAC: {:>.4f}\tREC: {:>.4f}'.format(lr_model_4_acc, lr_model_4_bac, lr_model_4_rec))
model_names.append('lr_model_4')
model_feature_counts.append(X4_train.shape[1])
model_feature_names.append(list(X4_train.columns))
model_acc_scores.append(lr_model_4_acc)
model_bac_scores.append(lr_model_4_bac)
model_rec_scores.append(lr_model_4_rec)


# ### Model 11: DTC with X4

# In[40]:


dt_model_4 = DTC(random_state=0)
dt_model_4.fit(X4_train, y_train)
dt_model_4_y_hat = dt_model_4.predict(X4_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(dt_model_4_y_hat, name='Predicted')
dt_model_4_cm = pd.crosstab(y_actual, y_predicted)
print(dt_model_4_cm, '\n')
dt_model_4_acc = accuracy_score(y_test, dt_model_4_y_hat)
dt_model_4_bac = balanced_accuracy_score(y_test, dt_model_4_y_hat)
dt_model_4_rec = recall_score(y_test, dt_model_4_y_hat)
print('ACC: {:>.4f}\tBAC: {:>.4f}\tREC: {:>.4f}'.format(dt_model_4_acc, dt_model_4_bac, dt_model_4_rec))
model_names.append('dt_model_4')
model_feature_counts.append(X4_train.shape[1])
model_feature_names.append(list(X4_train.columns))
model_acc_scores.append(dt_model_4_acc)
model_bac_scores.append(dt_model_4_bac)
model_rec_scores.append(dt_model_4_rec)


# ### Model 12: RFC with X4

# In[41]:


rf_model_4 = RFC(random_state=0)
rf_model_4.fit(X4_train, y_train)
rf_model_4_y_hat = rf_model_4.predict(X4_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(rf_model_4_y_hat, name='Predicted')
rf_model_4_cm = pd.crosstab(y_actual, y_predicted)
print(rf_model_4_cm, '\n')
rf_model_4_acc = accuracy_score(y_test, rf_model_4_y_hat)
rf_model_4_bac = balanced_accuracy_score(y_test, rf_model_4_y_hat)
rf_model_4_rec = recall_score(y_test, rf_model_4_y_hat)
print('ACC: {:>.4f}\tBAC: {:>.4f}\tREC: {:>.4f}'.format(rf_model_4_acc, rf_model_4_bac, rf_model_4_rec))
model_names.append('rf_model_4')
model_feature_counts.append(X4_train.shape[1])
model_feature_names.append(list(X4_train.columns))
model_acc_scores.append(rf_model_4_acc)
model_bac_scores.append(rf_model_4_bac)
model_rec_scores.append(rf_model_4_rec)


# In[42]:


df_model_eval = pd.DataFrame({'model': model_names, 'feature_count': model_feature_counts, 
                              'feature_names': model_feature_names, 'acc': model_acc_scores, 
                              'bac': model_bac_scores, 'rec': model_rec_scores})
df_model_eval.round(2)


# ### Model 13: LRC with X4 and manual hyper-parameter tuning for class_weight

# In[43]:


lr_model_5 = LRC(random_state=0, class_weight='balanced')
lr_model_5.fit(X4_train, y_train)
lr_model_5_y_hat = lr_model_5.predict(X4_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(lr_model_5_y_hat, name='Predicted')
lr_model_5_cm = pd.crosstab(y_actual, y_predicted)
lr_model_5_acc = accuracy_score(y_test, lr_model_5_y_hat)
lr_model_5_bac = balanced_accuracy_score(y_test, lr_model_5_y_hat)
lr_model_5_rec = recall_score(y_test, lr_model_5_y_hat)
model_names.append('lr_model_5')
model_feature_counts.append(X4_train.shape[1])
model_feature_names.append(list(X4_train.columns))
model_acc_scores.append(lr_model_5_acc)
model_bac_scores.append(lr_model_5_bac)
model_rec_scores.append(lr_model_5_rec)


# ### Model 14: LRC with X4 and automatic hyper-parameter tuning using GridSearchCV

# In[44]:


from sklearn.model_selection import GridSearchCV


# In[45]:


lr_hp_grid = {'C': [0.01, 0.1, 1.0, 2.0, 5.0], 
              'class_weight': [None, 'balanced'],
              'solver': ['lbfgs', 'liblinear', 'newton-cg']}


# In[46]:


lr_gscv = GridSearchCV(estimator=LRC(random_state=0), param_grid=lr_hp_grid, verbose=10)


# In[47]:


lr_gscv.fit(df_feature_set_4, labels)


# In[48]:


lr_gscv.best_params_


# In[49]:


lr_model_6 = LRC(random_state=0, class_weight=None, C=0.1, solver='liblinear')
lr_model_6.fit(X4_train, y_train)
lr_model_6_y_hat = lr_model_6.predict(X4_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(lr_model_6_y_hat, name='Predicted')
lr_model_6_cm = pd.crosstab(y_actual, y_predicted)
lr_model_6_acc = accuracy_score(y_test, lr_model_6_y_hat)
lr_model_6_bac = balanced_accuracy_score(y_test, lr_model_6_y_hat)
lr_model_6_rec = recall_score(y_test, lr_model_6_y_hat)
model_names.append('lr_model_6')
model_feature_counts.append(X4_train.shape[1])
model_feature_names.append(list(X4_train.columns))
model_acc_scores.append(lr_model_6_acc)
model_bac_scores.append(lr_model_6_bac)
model_rec_scores.append(lr_model_6_rec)


# In[50]:


df_model_eval = pd.DataFrame({'model': model_names, 'feature_count': model_feature_counts, 
                              'feature_names': model_feature_names, 'acc': model_acc_scores, 
                              'bac': model_bac_scores, 'rec': model_rec_scores})
df_model_eval.round(2)


# ### Model 15: LRC with X4 and automatic hyper-parameter tuning using GridSearchCV and specified evaluation metric (balanced_accuracy instead of accuracy in this case)

# In[51]:


lr_gscv_2 = GridSearchCV(estimator=LRC(random_state=0), param_grid=lr_hp_grid, verbose=10, scoring='balanced_accuracy')


# In[52]:


lr_gscv_2.fit(df_feature_set_4, labels)


# In[53]:


lr_gscv_2.best_params_


# In[54]:


lr_model_7 = LRC(random_state=0, class_weight='balanced', C=0.1, solver='liblinear')
lr_model_7.fit(X4_train, y_train)
lr_model_7_y_hat = lr_model_7.predict(X4_test)
y_actual = pd.Series(y_test.values, name='Actual')
y_predicted = pd.Series(lr_model_7_y_hat, name='Predicted')
lr_model_7_cm = pd.crosstab(y_actual, y_predicted)
lr_model_7_acc = accuracy_score(y_test, lr_model_7_y_hat)
lr_model_7_bac = balanced_accuracy_score(y_test, lr_model_7_y_hat)
lr_model_7_rec = recall_score(y_test, lr_model_7_y_hat)
model_names.append('lr_model_7')
model_feature_counts.append(X4_train.shape[1])
model_feature_names.append(list(X4_train.columns))
model_acc_scores.append(lr_model_7_acc)
model_bac_scores.append(lr_model_7_bac)
model_rec_scores.append(lr_model_7_rec)


# In[55]:


df_model_eval = pd.DataFrame({'model': model_names, 'feature_count': model_feature_counts, 
                              'feature_names': model_feature_names, 'acc': model_acc_scores, 
                              'bac': model_bac_scores, 'rec': model_rec_scores})
df_model_eval.round(2)


# ### Since we are considering BAC as the primary metric and REC as secondary metric for evaluating the models, let us assume the following weighted scoring:
# 
# 60% of BAC + 40% of REC

# In[56]:


df_model_eval['weighted_score'] = df_model_eval.bac * 0.6 + df_model_eval.rec * 0.4
df_model_eval.round(2)


# In[57]:


df_model_eval.sort_values(by='weighted_score', inplace=True)
df_model_eval.round(2)


# ### Always remember Ockham's Razor, a.k.a., The Law of Parsimony:
# 
# You will come across various explanations of this principle, but a simple way to understand the principle is "simplify when you can"!!
# 
# If you have to select the best model from the above 15 models, you can start by considering the top 5 models. Then compare them for the scores and parsimony.
# 
# From the above table, the top 3 based on score alone are lr_model_4, lr_model_3 and lr_model_6. Among these, the most parsimonious are lr_model_4 and lr_model_6. So, from both the score perspective and parsimony perspective, lr_model_4 is the best model.
# 
# Now, assume that lr_model_4 does not exist. In that case we are left with considering lr_model_3 and lr_model_6. Both have the same scores, but lr_model_6 is much more parsimonious. In this case, lr_model_6 would be the best model.
