#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats

accepts = pd.read_csv('accepts.csv')
cross_table = pd.crosstab(accepts.bankruptcy_ind, accepts.bad_ind, margins=True)
print(cross_table)

cross_table_rowpct = cross_table.div(cross_table["All"], axis=0)
print(cross_table_rowpct)

print(
    "chisq = %6.4f\np-value = %6.4f\ndof = %d\nexpected_freq = %s"
    % stats.chi2_contingency(cross_table))


# In[ ]:




