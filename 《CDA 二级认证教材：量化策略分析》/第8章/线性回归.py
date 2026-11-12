#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols

data_raw = pd.read_excel('LR_practice.xlsx')
print(data_raw.head())
print(data_raw.tail())
print(data_raw.head(2))
print(data_raw.info())

data_raw.drop(['id','Acc','edad2'], axis=1, inplace=True)
print(data_raw.head(2))

data_raw.drop_duplicates(inplace=True)

print(data_raw.isnull().mean())
print(data_raw.isnull().sum())
print(data_raw['avg_exp'].mean())

data_raw['avg_exp'] = data_raw['avg_exp'].fillna(data_raw['avg_exp'].mean())
print(data_raw.isnull().mean())

data_raw['gender'] = data_raw['gender'].map({'Male':1, 'Female':0})
print(data_raw.head(2))

print(data_raw['edu_class'].unique())
label = data_raw['edu_class'].unique().tolist()
print(label)
print(label.index('中学'))

data_raw['edu_class'] = data_raw['edu_class'].apply(lambda x: label.index(x))
print(data_raw.head())
print(data_raw['edu_class'].value_counts())

sns.boxplot(data=data_raw['dist_avg_income'])
plt.show()

z = stats.zscore(data_raw['Age'])
print(z)
print((z>3)|(z<-3))
print(z[(z>3)|(z<-3)])
print(data_raw['Age'].iloc[40])
print(data_raw['Age'].mean())
print(data_raw['Age'].drop(index=40).mean())

data_raw.loc[40, 'Age'] = data_raw['Age'].drop(index=40).mean()
print(data_raw.head())

dummy = pd.get_dummies(data=data_raw['edu_class'], prefix='edu', drop_first=True)
print(dummy)

data = pd.concat([data_raw, dummy], axis=1)
print(data.head())

print(data.corr())
print(data.corr(method='kendall'))

plt.scatter(data['Income'], data['avg_exp'])
plt.show()

print(data.columns)

LR1 = 'avg_exp~gender+Age+ Income+ Ownrent+ Selfempl+dist_home_val+ dist_avg_income+edu_1+ edu_2+edu_3+ edu_4'
model_1 = ols(LR1, data).fit()
print(model_1.summary())
print(dir(model_1))
print(model_1.resid)

print(data.head())

def vif(df, y_col):
    x_cols = list(df.columns)
    x_cols.remove(y_col)
    formula = y_col + '~' + '+'.join(x_cols)
    rsquared = ols(formula, df).fit().rsquared
    return 1. / (1. - rsquared)

exog = data[['gender','Age','Income','Ownrent','Selfempl','dist_home_val','dist_avg_income','edu_class']]
for y_col in exog.columns:
    print(y_col, '\t', vif(df=exog, y_col=y_col))

LR2 = 'avg_exp~gender+Age+ Income+ Ownrent+ Selfempl+dist_home_val+edu_1+ edu_2+edu_3+ edu_4'
model_2 = ols(LR2, data).fit()
print(model_2.summary())

plt.scatter(model_2.predict(data), model_2.resid)
plt.show()

res = stats.probplot(model_2.resid, plot=plt)
plt.show()

data['ln_avg_exp'] = np.log(data['avg_exp'])
print(data.head())

LR3 = 'ln_avg_exp~gender+Age+ Income+ Ownrent+ Selfempl+dist_home_val+edu_1+ edu_2+edu_3+ edu_4'
model_3 = ols(LR3, data).fit()
print(model_3.summary())

res = stats.probplot(model_3.resid, plot=plt)
plt.show()

data['Age_sq'] = data['Age']**2
print(data.head())

LR4 = 'ln_avg_exp~gender+Age+ Income+ Ownrent+ Selfempl+dist_home_val+edu_1+ edu_2+edu_3+ edu_4 + Age_sq'
model_4 = ols(LR4, data).fit()
print(model_4.summary())

data['Age_gender'] = data['Age']*data['gender']
print(data.head())

LR5 = 'ln_avg_exp~gender+Age+ Income+ Ownrent+ Selfempl+dist_home_val+edu_1+ edu_2+edu_3+ edu_4 + Age_sq+Age_gender'
model_5 = ols(LR5, data).fit()
print(model_5.summary())


# In[ ]:




