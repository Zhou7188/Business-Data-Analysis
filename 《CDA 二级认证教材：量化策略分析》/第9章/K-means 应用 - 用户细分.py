#!/usr/bin/env python
# coding: utf-8

# In[3]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.tree import plot_tree

model_data = pd.read_csv("profile_bank.csv")
data = model_data.loc[:, 'CNT_TBM':'CNT_CSC']
print(data.head())

corr_matrix = data.corr(method='pearson')
print(corr_matrix)

data = preprocessing.scale(data)

pca = PCA(n_components=3)
newData = pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting

fa = FactorAnalysis.load_data_samples(data, preproc_demean=True, preproc_scale=True)
fa.extract_components()

fa.find_comps_to_retain(method='top_n', num_keep=3)

fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)
plt.show()

print(pd.DataFrame(fa.comps["rot"]))

score = fa.get_component_scores(data)
score = pd.DataFrame(score)

fa_scores = score.rename(columns={0: "ATM_POS", 1: "TBM", 2: "CSC"})
print(fa_scores.head())

var = ["ATM_POS", "TBM", "CSC"]
skew_var = {}
for i in var:
    skew_var[i] = abs(fa_scores[i].skew())
skew = pd.Series(skew_var).sort_values(ascending=False)
print(skew)

kmeans = KMeans(n_clusters=3)
result = kmeans.fit(fa_scores)

model_data_l = model_data.join(pd.DataFrame(result.labels_))
model_data_l = model_data_l.rename(columns={0: "clustor"})
print(model_data_l.head())

model_data_l.clustor.value_counts().plot(kind="pie")
plt.show()

quantile_transformer = preprocessing.QuantileTransformer(
    output_distribution='normal', random_state=0
)
fa_scores_trans = quantile_transformer.fit_transform(fa_scores)
fa_scores_trans = pd.DataFrame(fa_scores_trans)
fa_scores_trans = fa_scores_trans.rename(
    columns={0: "ATM_POS", 1: "TBM", 2: "CSC"}
)

var = ["ATM_POS", "TBM", "CSC"]
skew_var = {}
for i in var:
    skew_var[i] = abs(fa_scores_trans[i].skew())
skew = pd.Series(skew_var).sort_values(ascending=False)
print(skew)

kmeans = KMeans(n_clusters=3)
result = kmeans.fit(fa_scores_trans)

model_data_l = model_data.join(pd.DataFrame(result.labels_))
model_data_l = model_data_l.rename(columns={0: "clustor"})
print(model_data_l.head())

print(model_data_l.clustor.value_counts())

clf = tree.DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_split=100,
    min_samples_leaf=100,
    random_state=12345,
)
clf.fit(model_data, result.labels_)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(15, 10))
plot_tree(clf, 
          feature_names=model_data.columns, 
          class_names=["0", "1", "2"], 
          filled=True)
plt.show()


# In[ ]:




