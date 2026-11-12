#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r"2017股价.xlsx")
print(df)

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure(figsize=(10, 4))
plt.plot(range(15), df.loc[:, "收盘价"], '-*')
plt.plot(range(15), df.loc[:, "开盘价"], '-o')
plt.plot(range(15), df.loc[:, "最高价"], '-v')
plt.plot(range(15), df.loc[:, "最低价"], '-^')
plt.title('股价走势图')
plt.xlabel('时间')
plt.ylabel('股价')
plt.xticks(range(15), df.日期, rotation=30)
plt.legend(['收盘价', '开盘价', '最高价', '最低价'])
plt.show()

np.random.seed(1)
df = pd.DataFrame(data={"方案": ["plan A", "plan B", "plan C", "plan D", "plan E"],
                        "投票人数": np.random.randint(10, 100, 5)})
print(df)

plt.figure(figsize=(6, 6))
plt.pie(df.投票人数, labels=df.方案, autopct='%.2f%%', textprops={'fontsize': 13})
plt.title('方案投票占比饼图', fontsize=14)
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(df.方案, df.投票人数, width=0.6)
plt.title("各方案投票人数柱形图", fontsize=14)
plt.ylabel("投票人数", fontsize=13)
plt.xlabel("方案", fontsize=13)
plt.show()

plt.figure(figsize=(8, 4))
plt.barh(df.sort_values("投票人数").方案, df.sort_values("投票人数").投票人数, height=0.6)
plt.xlabel("投票人数", fontsize=13)
plt.ylabel("方案", fontsize=13)
plt.title("各方案投票人数条形图", fontsize=14)
plt.show()

np.random.seed(1)
data = np.random.normal(loc=80, scale=20, size=200).astype(int)
print(data)

plt.hist(data, [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
plt.xticks([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
plt.show()

df = pd.DataFrame({"产量": [2200, 3300, 4400, 5500, 5700, 7300, 8000],
                   "降水量": [22, 39, 55, 66, 105, 100, 122]})
print(df)

plt.scatter(df.降水量, df.产量)
plt.show()

np.random.seed(1)
df = pd.DataFrame({"语文": np.random.randint(60, 150, 10),
                   "英语": np.random.randint(40, 150, 10),
                   "数学": np.random.randint(20, 150, 10)})
print(df)

plt.figure(figsize=(8, 6))
plt.boxplot([df.语文, df.英语, df.数学], labels=['语文', '英语', '数学'])
plt.show()

np.random.seed(1)
data = np.random.normal(loc=80, scale=20, size=200).astype(int)
print(data)

sns.displot(data, bins=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130], kde=True)
plt.show()

np.random.seed(0)
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
print(df.head())

sns.jointplot(x="x", y="y", data=df)
plt.show()

sns.relplot(x="x", y="y", data=df)
plt.show()

sns.jointplot(x="x", y="y", data=df, kind="kde")
plt.show()

sns.displot(data=df, x="x", y="y", kind="kde")
plt.show()

np.random.seed(0)
df0 = pd.DataFrame(np.random.uniform(-1, 1, (7, 7)), index=list("abcdefg"), columns=list("ABCDEFG"))
print(df0)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.5)
sns.heatmap(df0)
plt.show()


# In[ ]:




