#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 8)

cities = pd.read_csv("cities_10.csv", encoding='gbk')
cities.head()

cities.loc[:, 'X1':].corr(method='pearson').head()

from sklearn.preprocessing import StandardScaler

n_components = 2

scaler = StandardScaler()
scale_cities = scaler.fit_transform(cities.loc[:, 'X1':])

pca = PCA(n_components=n_components)
pca_scores = pca.fit_transform(scale_cities)
print('Cumsum of explained variance ratio: ',
      pca.explained_variance_ratio_.cumsum())

pca_scores = pd.DataFrame(pca_scores, columns=['Gross', 'Avg'])
scale_pca_cities = cities.join(pca_scores)
scale_pca_cities.head()

print(pca.components_)

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def scatter(x, y, labels, **kw):
    plt.figure(figsize=(6, 4))
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.scatter(x, y)
    for xx, yy, label in zip(x, y, labels):
        plt.text(xx, yy, '%s.' % label,
                 ha='center', va='bottom', **kw)
    plt.show()

x = scale_pca_cities['Gross']
y = scale_pca_cities['Avg']
labels = scale_pca_cities['AREA']
scatter(x, y, labels, fontsize=15)

from sklearn.decomposition import SparsePCA
alpha = 0.5
spca = SparsePCA(n_components=n_components, alpha=alpha)
spca.fit(scale_cities)
pd.DataFrame(spca.components_)

from functools import reduce

n_cols = scale_cities.shape[1]
spca_with_alpha = []

for i in np.arange(0.1, 1, 0.05):
    spca = SparsePCA(n_components=n_components, alpha=i)
    spca.fit(scale_cities)
    components_ = pd.DataFrame(spca.components_).T
    diff = np.count_nonzero(components_) - n_cols
    spca_with_alpha.append((i, diff))

selector = pd.DataFrame(spca_with_alpha)
selector.T

spca_ = SparsePCA(n_components=n_components, alpha=0.65).fit(scale_cities)
pd.DataFrame(spca_.components_)

scale_spca_cities = pd.DataFrame(spca_.transform(scale_cities), columns=['Gross', 'Avg'])
scale_spca_cities

x = scale_spca_cities['Gross']
y = scale_spca_cities['Avg']
label = cities['AREA']

scatter(x, y, label)

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting

fa = FactorAnalysis.load_data_samples(
        scale_cities,
        preproc_demean=True,
        preproc_scale=True
        )
fa.extract_components()

fa.find_comps_to_retain(method='top_n',num_keep=2)

from pylab import mpl

mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)

plt.show()

print(fa.comps['rot'])

fa_scores = fa.get_component_scores(scale_cities)

cities_factor_analysis = cities.join(
    pd.DataFrame(fa_scores, columns=['Gross', 'Avg']))
cities_factor_analysis.to_csv('cities_FA_score.csv', index=False)
cities_factor_analysis.head()

x = cities_factor_analysis['Gross']
y = cities_factor_analysis['Avg']
label = cities_factor_analysis['AREA']

scatter(x, y, label,fontsize=15)

import scipy.cluster.hierarchy as sch

distMat = sch.distance.pdist(
    cities_factor_analysis[['Gross','Avg']],'euclidean')

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

cities_factor_analysis.loc[:, 'X1':'X9'].groupby(
    cities_factor_analysis['cluster']).mean().T


# In[ ]:




