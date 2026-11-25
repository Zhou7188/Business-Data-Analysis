#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.set_option('display.max_columns', 8)

raw_data = pd.read_csv('creditcard_exp.csv', skipinitialspace=True)
out_of_sample_data = pd.read_csv('creditcard_exp_out_of_sample.csv', skipinitialspace=True)
print(raw_data.head())

train = raw_data[raw_data['avg_exp'].notnull()].copy().iloc[:, 2:].drop('age2', axis=1)
oos_data = out_of_sample_data.copy().drop('age2', axis=1)

print(train.describe(include='all'))

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel('avg_exp', fontsize=15)
ax.set_xlabel('Income', fontsize=15)
train.plot('Income', 'avg_exp', kind='scatter', ax=ax)
plt.show()

print(train[['Income', 'avg_exp', 'Age', 'dist_home_val']].corr(method='pearson'))

simple_linear_model = ols(formula='avg_exp ~ Income', data=train).fit()
print(simple_linear_model.params)

summary = simple_linear_model.summary()
for table in summary.tables:
    print(table)

print(simple_linear_model.summary())

train_prediction = simple_linear_model.predict(train)
train_resid = simple_linear_model.resid
print(pd.DataFrame([train_prediction, train_resid], index=['Pred_avg_exp', 'resid']).T.head())

print(simple_linear_model.predict(oos_data)[:5])

multiple_linear_model = ols('avg_exp ~ Income + dist_home_val + dist_avg_income', data=train).fit()
print(multiple_linear_model.summary())

print(multiple_linear_model.params)

def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response, ' + '.join(selected + [candidate]))
            aic = ols(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = aic_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print('aic is {},continuing!'.format(current_score))
        else:
            print('forward selection over!')
            break

    formula = "{} ~ {} ".format(response, ' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = ols(formula=formula, data=data).fit()
    return model

data_for_select = train[['avg_exp', 'Income', 'Age', 'dist_home_val', 'dist_avg_income']]
forward_select_model = forward_select(data=data_for_select, response='avg_exp')
print(forward_select_model.rsquared)

def plot_resid(resid, x, y_hat, **kw):
    plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(211)
    ax1.set_ylabel('resid', fontsize=15)
    ax1.set_xlabel('x', fontsize=15)
    ax1.scatter(x, resid, **kw)
    ax1.set_title('Scatter of resid - x', fontsize=15)

    ax2 = plt.subplot(212)
    ax2.set_ylabel('resid', fontsize=15)
    ax2.set_xlabel('y_hat', fontsize=15)
    ax2.scatter(y_hat, resid, **kw)
    ax2.set_title('Scatter of resid - y_hat', fontsize=15)

    plt.tight_layout() 
    plt.show()

train['Income_ln'] = np.log(train['Income'])
oos_data['Income_ln'] = np.log(oos_data['Income'])

linear_model1 = ols(formula='avg_exp ~ Income', data=train).fit()
linear_model2 = ols(formula='avg_exp_ln ~ Income', data=train).fit()
linear_model3 = ols(formula='avg_exp_ln ~ Income_ln', data=train).fit()


# In[ ]:




