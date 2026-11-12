#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

churn = pd.read_csv('telecom_churn.csv')   
print(churn.head())
print(churn.info())

cross_table = pd.crosstab(churn.posTrend, churn.churn, margins=True)
print(cross_table)

def percConvert(ser):
    return ser/float(ser[-1])

print(cross_table.apply(percConvert, axis=1))
print(544/1644)

print('''chisq = %6.4f 
p-value = %6.4f
dof = %i 
expected_freq = %s'''  % stats.chi2_contingency(cross_table.iloc[:2, :2]))

churn.plot(x='duration', y='churn', kind='scatter')
plt.show()

train = churn.sample(frac=0.7, random_state=666)
test = churn[~ churn.index.isin(train.index)]
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))

lg = smf.logit('churn ~ duration', train).fit()
print(lg.summary())

train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)

print(test[['churn','proba']].head())

test['prediction'] = (test['proba'] > 0.3).astype('int')
print(test[['churn','prediction']].head(10))
print(pd.crosstab(test.churn, test.prediction, margins=True))

acc = sum(test['prediction'] == test['churn']) / len(test)
print('The accurancy is %.2f' % acc)

for i in np.arange(0.1, 0.9, 0.1):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction, test.churn, margins=True)
    precision = confusion_matrix.loc[0, 0] / confusion_matrix.loc['All', 0]
    recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All']
    Specificity = confusion_matrix.loc[1, 1] / confusion_matrix.loc[1, 'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f' % (i, precision, recall, Specificity, f1_score))

fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

print('AUC = %.4f' % metrics.auc(fpr_test, tpr_test))
print(th_test[(tpr_test - fpr_test).argmax()])

def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.logit(
                formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break

    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.logit(formula=formula, data=data).fit()
    return(model)

candidates = ['churn','duration','AGE','incomeCode','nrProm','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[candidates]

lg_m1 = forward_select(data=data_for_select, response='churn')
print(lg_m1.summary())

print(f'原来有 {len(candidates)-1} 个变量')
print(f'筛选剩下 {len(lg_m1.params.index.tolist())} 个(包含 intercept 截距项)。')

class_col = ['posTrend','prom','edu_class','feton','curPlan','avgplan','gender','negTrend','planChange','posPlanChange','negPlanChange']
for i in class_col:
    tab = pd.crosstab(churn[i], churn.churn)
    print(i,''' p-value = %6.4f''' % stats.chi2_contingency(tab)[1])

formula = '''churn ~ duration + AGE +  incomeCode + peakMinDiff + call_10086 +C(edu_class) +C(feton)+ C(posTrend) + C(negTrend) + C(curPlan) + C(avgplan) + C(gender)'''
lg_m = smf.logit(formula=formula, data=train).fit()
print(lg_m.summary())

train['proba'] = lg_m.predict(train)
test['proba'] = lg_m.predict(test)
print(test[['churn','proba']].head(10))

test['prediction'] = (test['proba'] > 0.5).astype('int')
print(pd.crosstab(test.churn, test.prediction, margins=True))

acc_m = sum(test['prediction'] == test['churn']) /len(test)
print('The accurancy is %.2f' %acc_m)

for i in np.arange(0.1, 0.9, 0.1):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.churn, margins = True)
    precision = confusion_matrix.loc[0, 0] /confusion_matrix.loc['All', 0]
    recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All']
    Specificity = confusion_matrix.loc[1, 1] /confusion_matrix.loc[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))

fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()

print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))
print(th_test[(tpr_test - fpr_test).argmax()])

test['prediction'] = (test['proba'] > 0.456).astype('int')
print(pd.crosstab(test.churn, test.prediction, margins=True))

def vif(df, col_i):
    from statsmodels.formula.api import ols
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

exog = train[candidates].drop(['churn'], axis=1)
for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))


# In[ ]:




