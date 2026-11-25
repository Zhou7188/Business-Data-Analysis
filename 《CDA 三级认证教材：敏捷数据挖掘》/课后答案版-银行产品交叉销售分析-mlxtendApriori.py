#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

file_path = 'bank_prod_order(1).csv'
data = pd.read_csv(file_path, encoding='gbk')
data.head()

grouped = data.groupby('CSR_ID')['PROD'].apply(list)

te = TransactionEncoder()
te_ary = te.fit(grouped).transform(grouped)
df = pd.DataFrame(te_ary, columns=te.columns_)

def encode_units(x):
    if x ==True:
        return 1
    else:
        return 0

df=df.applymap(encode_units)
df.head()

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules

conf = rules[(rules.confidence > 0.1) & (rules.lift > 1)]
conf.sort_values(by='support', ascending=False).head()

import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_xlabel('lift')
conf.plot.scatter(x="lift", y="support",c="confidence",cmap='binary',
                  figsize=(6,4), ax=ax)

plt.show()

rules

rules['antecedents'] = rules['antecedents'].astype(str)
rules['consequents'] = rules['consequents'].astype(str)

caiwutong_rules = rules[rules['antecedents'].str.contains('财务通') | rules['consequents'].str.contains('财务通')]

print("与'财务通'相关的规则:")
print(caiwutong_rules)

caiwutong_rules

haoxiangdai_rules = rules[rules['consequents'].str.contains('好享贷')]

print("与'好享贷'相关的规则:")
haoxiangdai_rules


# In[ ]:




