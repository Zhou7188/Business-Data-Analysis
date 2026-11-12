#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from fa_kit import FactorAnalysis
from pylab import mpl

model_data = pd.read_csv("cities_10.csv", encoding='gbk')
print(model_data.head())

data = model_data.loc[:, 'X1':]
print(data.head())

data = preprocessing.scale(data)
print(data)

pca = PCA(n_components=9)
pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

pca = PCA(n_components=2).fit(data)
print(pca.explained_variance_)

fa = FactorAnalysis.load_data_samples(
    data,
    preproc_demean=True,
    preproc_scale=True
)
fa.extract_components()

print(fa.find_comps_to_retain(method='top_n', num_keep=2))

fa.rotate_components(method='varimax')
print(pd.DataFrame(fa.comps["rot"]))

fas = pd.DataFrame(fa.comps["rot"])
data = pd.DataFrame(data)
fa_score = pd.DataFrame(np.dot(data, fas))
print(data)
print(fas)
print(fa_score)

a = fa_score.rename(columns={0: "Gross", 1: "Avg"})
citi10_fa = model_data.join(a)
print(citi10_fa)

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x = citi10_fa['Gross']
y = citi10_fa['Avg']
label = citi10_fa['AREA']
plt.scatter(x, y)
for a, b, l in zip(x, y, label):
    plt.text(a, b + 0.1, '%s.' % l, ha='center', va='bottom', fontsize=14)
plt.show()


# In[ ]:




