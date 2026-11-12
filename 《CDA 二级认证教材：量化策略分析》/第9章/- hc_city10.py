#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("cities_10.csv", encoding="gbk")
print(df)

model_data = df.loc[:, "X1":]
print(model_data)

corr_matrix = model_data.corr()
print(corr_matrix)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler_model_data = scaler.fit_transform(model_data)
print(scaler_model_data)

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
model_data_scale = pca.fit(scaler_model_data)
print(model_data_scale)

np.set_printoptions(suppress=True)
print("主成分可解释的变异：\n", pca.explained_variance_)
print("主成分可解释的变异的比例:\n", pca.explained_variance_ratio_)

print(0.8 + 0.12)

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting

fa = FactorAnalysis.load_data_samples(
    scaler_model_data, preproc_demean=True, preproc_scale=True
)
fa.extract_components()

print(fa.find_comps_to_retain(method="top_n", num_keep=2))

fa.rotate_components(method="varimax")

fa_plotting.graph_summary(fa)
plt.show()

print(fa.comps["rot"])

fa_scores = fa.get_component_scores(scaler_model_data)
fa_scores = pd.DataFrame(fa_scores, columns=["Gross", "Avg"])
print(fa_scores)

cities_fa_scores = df.join(fa_scores)
print(cities_fa_scores)

cities_fa_scores.to_csv("fa_cities_10.csv", index=False)

new_model_data = pd.read_csv("fa_cities_10.csv")
print(new_model_data)


def scatter(x, y, labels, **kw):
    plt.figure(figsize=(6, 4))
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.scatter(x, y)
    for xx, yy, label in zip(x, y, labels):
        plt.text(xx, yy, '%s.' % label, ha='center', va='bottom', **kw)
    plt.show()


x = new_model_data["Gross"]
y = new_model_data["Avg"]
labels = new_model_data["AREA"]
scatter(x, y, labels)

import scipy.cluster.hierarchy as sch

dis_mat = sch.distance.pdist(
    new_model_data[["Gross", "Avg"]], metric='euclidean'
)

Z = sch.linkage(dis_mat, method="ward")
print(Z)

P = sch.dendrogram(Z, labels=new_model_data["AREA"].tolist())
plt.savefig("plot_dendrogram.png")
plt.show()


# In[ ]:




