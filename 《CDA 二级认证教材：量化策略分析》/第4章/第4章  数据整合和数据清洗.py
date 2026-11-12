#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv('sample.csv')
print(csv)

csv = pd.read_csv('sample.csv', usecols=['id', 'name'], nrows=2)
print(csv)

csvs = pd.read_csv('data/big.csv', chunksize=900)
for i in csvs:
    print(i.shape)

csvs = pd.read_csv('data/big.csv', chunksize=900)
dat = pd.concat(csvs, ignore_index=True)
print(dat.shape)

csv = pd.read_csv('sample.csv', na_values='99999')
print(csv)

csv = pd.read_csv('sample.csv', encoding='utf-8')
print(csv)

csv.to_csv('write.csv', encoding='utf-8', index=False)

np.random.seed(0)
sample = pd.DataFrame(np.random.randn(4, 5), columns=list("abcde"))
print(sample)

print(sample['a'])
print(sample.iloc[:, 0])
print(sample.loc[:, "a"])
print(sample[['a']])
print(sample.loc[:1, :"b"])
print(sample.iloc[0:2, 0:2])

sample['new_col'] = sample['a'] - sample['b']
print(sample)

print(sample.assign(new_col2=sample['a'] - sample['b'], new_col3=sample['a'] + sample['b']))
print(sample.drop(columns='a'))
print(sample.drop(columns=['a', 'b']))

sample = pd.DataFrame({
    'group': [1, 1, 1, 2, 1, 2],
    'name': ['Bob', 'Lindy', 'Mark', 'Miki', 'Sully', 'Rose'],
    'score': [98, 78, 87, 77, 65, 67]
})
print(sample)

print(sample.score > 70)
print(sample[sample.score > 70])
print(sample[(sample.score > 70) & (sample.group == 1)])
print(sample[~(sample.group == 1)])
print(sample[(sample.group == 2) | (sample.group == 1)])
print(sample.query('score > 90'))
print(sample.query('(group == 2) | (group == 1)'))
print(sample[sample['score'].between(70, 80, inclusive='both')])
print(sample[sample['name'].isin(['Bob', 'Lindy'])])
print(sample[sample['name'].str.contains('[M]+')])

df1 = pd.DataFrame({'col1': ['a', 'b', 'c'], 'id': [1, 2, 3]})
df2 = pd.DataFrame({'col2': ['d', 'e'], 'id': [4, 3]})
print(df1)
print(df2)
print(df1.merge(df2, how='inner', on='id'))
print(df1.merge(df2, how='inner', left_on='id', right_on='id'))
print(df1.merge(df2, how='left', on='id'))
print(df1.merge(df2, how='right', on='id'))
print(df1.merge(df2, how='outer', on='id'))

df1 = pd.DataFrame({'id1': [1, 2, 3], 'col1': ['a', 'b', 'c']}, index=[1, 2, 3])
df2 = pd.DataFrame({'id2': [1, 2, 3], 'col2': ['aa', 'bb', 'cc']}, index=[1, 3, 2])
print(df1)
print(df2)
print(pd.concat([df1, df2], axis=1))
print(df1.join(df2))

df1 = pd.DataFrame({'id': [1, 1, 1, 2, 3, 4, 6], 'col': ['a', 'a', 'b', 'c', 'v', 'e', 'q']})
df2 = pd.DataFrame({'id': [1, 2, 3, 3, 5], 'col': ['x', 'y', 'z', 'v', 'w']})
print(df1)
print(df2)
print(pd.concat([df1, df2], ignore_index=True, axis=0))
print(pd.concat([df1, df2], ignore_index=True).drop_duplicates())

df3 = df1.rename(columns={'col': 'new_col'})
print(df3)
print(pd.concat([df1, df3], ignore_index=True).drop_duplicates())

sample = pd.DataFrame({
    'group': [1, 1, 1, 2, 1, 2],
    'name': ['Bob', 'Lindy', 'Mark', 'Miki', 'Sully', 'Rose'],
    'score': [98, 78, 87, 77, 77, np.nan]
})
print(sample)
print(sample.sort_values('score', ascending=False, na_position='last'))
print(sample.sort_values(['group', 'score']))

sample = pd.DataFrame({
    "name": ['Bob', 'Lindy', 'Mark', 'Miki', 'Sully', 'Rose'],
    "class": [1, 1, 1, 2, 1, 2],
    "grade": [1, 1, 1, 2, 2, 2],
    "chinese": [88, 78, 86, 56, 77, 54],
    "math": [98, 78, 87, 77, 77, np.nan]
})
print(sample)
print(sample.groupby('grade')[['math']].max())
print(sample.groupby(['grade', 'class'])[['math']].mean())
print(sample.groupby(['grade'])[['math', 'chinese']].mean())
print(sample.groupby('class')['math'].agg(['mean', 'min', 'max']))

df = sample.groupby(['grade', 'class'])[['math', 'chinese']].agg(['min', 'max'])
print(df)
print(df['math']['min'])
print(df.loc[:, ('math', 'min')])
print(df.loc[(2, 2), ('math', 'min')])

table = pd.DataFrame({
    'Monetary': [3608, 420, 1894, 3503, 4567],
    'cust_id': [10001, 10001, 10002, 10002, 10003],
    'type': ['Normal', 'Special_offer', 'Normal', 'Special_offer', 'Special_offer']
})
print(table)
print(pd.pivot_table(table, index='cust_id', columns='type', values='Monetary'))
print(pd.pivot_table(table, index='cust_id', columns='type', values='Monetary', fill_value=0, aggfunc='sum'))

sample = pd.DataFrame({
    'name': ['Bob', 'Lindy', 'Mark', 'Miki', 'Sully', 'Rose'],
    'score': [99, 78, 999, 77, 77, np.nan],
    'group': [1, 1, 1, 2, 1, 2]
})
print(sample)
print(sample.score.replace(999, np.nan))
print(sample.replace({'score': {999: np.nan}, 'name': {'Bob': np.nan}}))
print(sample)

def transform(row):
    if row['group'] == 1:
        return 'class1'
    elif row['group'] == 2:
        return 'class2'

print(sample.apply(transform, axis=1))
print(sample.assign(class_n=sample.apply(transform, axis=1)))

sample = sample.copy()
sample.loc[sample.group == 1, 'class_n'] = 'class1'
sample.loc[sample.group == 2, 'class_n'] = 'class2'
print(sample)

sample = pd.DataFrame({
    'group': [1, 1, 1, 2, 1, 2],
    'id': [1, 1, 1, 3, 4, 5],
    'name': ['Bob', 'Bob', 'Mark', 'Miki', 'Sully', 'Rose'],
    'score': [99, 99, 87, 77, 77, np.nan]
})
print(sample)
print(sample[sample.duplicated()])
print(sample.drop_duplicates())
print(sample.drop_duplicates('id'))

sample = pd.DataFrame({
    'group': [1, 1, np.nan, 2, 1, np.nan],
    'id': [1, 1, 1, 3, 4, np.nan],
    'name': ['Bob', 'Bob', 'Mark', 'Miki', 'Sully', np.nan],
    'score': [99, np.nan, 87, 77, 77, np.nan]
})
print(sample)
print(sample.apply(lambda col: sum(col.isnull()) / col.size))
print(sample.score.fillna(sample.score.mean()))
print(sample.score.fillna(sample.score.median()))
print(sample.score.isnull())
print(sample.score.isnull().apply(int))

def cap(x, quantile=[0.01, 0.99]):
    Q01, Q99 = x.quantile(quantile).values.tolist()
    if Q01 > x.min():
        x = x.copy()
        x.loc[x < Q01] = Q01
    if Q99 < x.max():
        x = x.copy()
        x.loc[x > Q99] = Q99
    return x

sample = pd.DataFrame({'normal': np.random.randn(1000)})
sample.hist(bins=50)

new = sample.apply(cap, quantile=[0.01, 0.99])
new.hist(bins=50)

np.random.seed(0)
sample = pd.DataFrame({'normal': np.random.randn(10)})
print(sample)
print(pd.cut(sample.normal, 5))
print(pd.cut(sample.normal, bins=5, labels=[1, 2, 3, 4, 5]))
print(pd.cut(sample.normal, bins=2, labels=['bad', 'good']))
print(sample.normal.quantile([0, 0.5, 1]))
print(pd.cut(sample.normal, bins=sample.normal.quantile([0, 0.5, 1]), include_lowest=True))
print(pd.cut(sample.normal, bins=sample.normal.quantile([0, 0.5, 1]), labels=['bad', 'good'], include_lowest=True))

plt.show()


# In[ ]:




