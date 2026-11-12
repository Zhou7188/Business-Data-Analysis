#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

camp = pd.read_csv(r"teleco_camp_orig.csv")

plt.hist(camp["AvgIncome"], bins=20)
plt.show()

print(camp["AvgIncome"].describe(include="all"))
print(camp["AvgHomeValue"].describe(include="all"))

camp["AvgIncome"] = camp["AvgIncome"].replace({0: np.nan})

plt.hist(camp["AvgIncome"], bins=20, range=(camp.AvgIncome.min(), camp.AvgIncome.max()))
plt.show()

print(camp["AvgHomeValue"].describe(include="all"))


# In[ ]:




