#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
sys.path.append("./myscript")

import pandas as pd
from varSelec import var_select

model_data = pd.read_csv("cities_10.csv", encoding='gbk')
print(model_data.head())

data = model_data.loc[ :,'X1':]

selected_data = var_select(data, k=2)
print(selected_data)


# In[ ]:





# In[ ]:





# In[ ]:




