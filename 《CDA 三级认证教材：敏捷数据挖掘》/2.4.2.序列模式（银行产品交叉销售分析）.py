#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv('bank_prod_order.csv', encoding='gbk')

baskets = data.groupby(['OrderNumber', 'CSR_ID'])['PROD'].apply(list)
baskets.head()

baskets = baskets.groupby(level=1).apply(list)
baskets.head()

import sys
sys.path.append('./myscript')
from aprioriAll import aprioriAll

patterns = aprioriAll(baskets, min_support=0.1)
patterns.head()

import sys

sys.path.append('./myscript')

from prefixspan import SquencePattern, prefixSpan

patterns = prefixSpan(SquencePattern([], 1e4), baskets, threshold=100)
for pattern in patterns:
    print(pattern.squence, "-> support:", pattern.support)


# In[ ]:




