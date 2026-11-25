#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv('bank_prod_order.csv',encoding='gbk')
data.head()

baskets = data['PROD'].groupby(data['CSR_ID']).apply(list)
baskets.head()

import sys
sys.path.append('./myscript')
from apriori import arules

rules = arules(baskets, min_support=0.02)
rules.head()

conf = rules[(rules.confidence > 0.1) & (rules.lift > 1)]
conf.sort_values(by='support', ascending=False).head()

import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_xlabel('lift')
conf.plot.scatter(x="lift", y="support",c="confidence",cmap='binary',
                  figsize=(6,4), ax=ax)

plt.show()

mapping = data['PROD'].drop_duplicates().to_dict()
inverse_mapping = {v:k for k, v in mapping.items()}

data['PROD'] = data['PROD'].replace(inverse_mapping)
baskets = data['PROD'].groupby(data['CSR_ID']).apply(list)  
baskets.head()

from orangecontrib.associate.fpgrowth import *  

n_examples=len(data)
itemsets = dict(frequent_itemsets(baskets.tolist(), 0.02))
rules = association_rules(itemsets, min_confidence=0.1)
stats = rules_stats(rules, itemsets, n_examples)

result = pd.DataFrame(stats, columns=['atecedent', 'consequent', 'support', 
                                      'confidence', 'coverage', 'strength', 
                                      'lift', 'leverage'])

pid_to_product = lambda fsets: tuple(mapping[fset] for fset in fsets)
result['atecedent'] = result['atecedent'].map(pid_to_product)
result['consequent'] = result['consequent'].map(pid_to_product)
result.head()


# In[ ]:




