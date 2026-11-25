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


grouped


te = TransactionEncoder()
te_ary = te.fit(grouped).transform(grouped)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


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


# In[ ]:




