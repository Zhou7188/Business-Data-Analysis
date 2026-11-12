#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

profile = pd.read_csv("profile_telecom.csv")
print(profile.head())

data = profile.loc[:, "cnt_call":]
print(data.corr(method="pearson"))

data_scaled = scale(data)
pca = PCA(n_components=2, whiten=True).fit(data_scaled)

print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.transform(data_scaled))


# In[ ]:




