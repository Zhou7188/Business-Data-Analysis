#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('sndHsPr.csv')

print(data.price.agg(['mean', 'median', 'std']))

print(data.price.quantile([0.25, 0.5, 0.75]))

data.price.hist(bins=20)
plt.show()

print(data.price.max() - data.price.min())

print(data.price.var())

print(data.price.std())

normal = pd.Series(np.random.randn(1000), name='normal')

print(normal.skew())

print(normal.kurtosis())

print(data[['AREA', 'price']].corr("pearson"))

print(data[['AREA', 'price']].corr("spearman"))

print(data[['AREA', 'price']].corr("kendall"))


# In[ ]:




