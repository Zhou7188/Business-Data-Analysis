#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fa_kit import FactorAnalysis

cities = pd.read_csv("cities_10.csv", encoding='gbk')

scaler = StandardScaler()
scale_cities = scaler.fit_transform(cities.loc[:, 'X1':])

fa = FactorAnalysis.load_data_samples(
    scale_cities,
    preproc_demean=True,
    preproc_scale=True
)
fa.extract_components()
fa.find_comps_to_retain(method='top_n', num_keep=2)
fa.rotate_components(method='varimax')
fa_scores = fa.get_component_scores(scale_cities)

cities_factor_analysis = cities.join(
    pd.DataFrame(fa_scores, columns=['Gross', 'Avg']))
cities_factor_analysis.to_csv('cities_FA_score.csv', index=False)

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

distMat = sch.distance.pdist(
    cities_factor_analysis[['Gross', 'Avg']], 'euclidean')

Z = sch.linkage(distMat, method='ward')

fig, ax = plt.subplots(figsize=(6, 4))
P = sch.dendrogram(
    Z,
    ax=ax,
    labels=cities_factor_analysis['AREA'].tolist(),
    leaf_font_size=15,
    leaf_rotation=90
)

plt.savefig('plot_dendrogram1.png')
plt.show()

labels = sch.cut_tree(Z, n_clusters=2)
cities_factor_analysis['cluster'] = labels

print(cities_factor_analysis.loc[:, 'X1':'X9'].groupby(
    cities_factor_analysis['cluster']).mean().T)


# In[ ]:




