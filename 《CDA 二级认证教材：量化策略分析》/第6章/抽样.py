#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np
import math


def get_sample(df, sampling="simple_random", k=1, stratified_col=None):
    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")
    elif k >= 1:
        assert isinstance(k, int), "选择抽样个数时, k必须为正整数"
        sample_by_n = True
        if sampling == "stratified":
            alln = k * df.groupby(by=stratified_col)[stratified_col[0]].count().count()
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
    else:
        sample_by_n = False
        if sampling in ("simple_random", "systematic"):
            k = math.ceil(len_df * k)

    if sampling == "simple_random":
        print("使用简单随机抽样")
        idx = random.sample(range(len_df), k)
        res_df = df.iloc[idx, :].copy()
        return res_df

    elif sampling == "systematic":
        print("使用系统抽样")
        step = len_df // k + 1
        start = 0
        idx = range(len_df)[start::step]
        res_df = df.iloc[idx, :].copy()
        return res_df

    elif sampling == "stratified":
        assert stratified_col is not None, "请传入包含需要分层的列名的列表"
        assert all(np.in1d(stratified_col, df.columns)), "请检查输入的列名"

        grouped = df.groupby(by=stratified_col)[stratified_col[0]].count()
        if sample_by_n == True:
            group_k = grouped.map(lambda x: k)
        else:
            group_k = grouped.map(lambda x: math.ceil(x * k))

        res_df = pd.DataFrame(columns=df.columns)
        for df_idx in group_k.index:
            df1 = df
            if len(stratified_col) == 1:
                df1 = df1[df1[stratified_col[0]] == df_idx]
            else:
                for i in range(len(df_idx)):
                    df1 = df1[df1[stratified_col[i]] == df_idx[i]]
            idx = random.sample(range(len(df1)), group_k[df_idx])
            group_df = df1.iloc[idx, :].copy()
            res_df = pd.concat([res_df, group_df])
        return res_df

    else:
        raise AssertionError("sampling is illegal")


clients = pd.read_csv(r'clients.csv', encoding='gbk')
print(clients.count())

srn = get_sample(clients, sampling="simple_random", k=22, stratified_col=None)
srp = get_sample(clients, sampling="simple_random", k=0.1, stratified_col=None)
print(srp.count())

strn = get_sample(clients, sampling="stratified", k=2, stratified_col=['Dist'])
strp = get_sample(clients, sampling="stratified", k=0.1, stratified_col=['Dist'])
print(strp.count())

sysn = get_sample(clients, sampling="systematic", k=4, stratified_col=None)
sysp = get_sample(clients, sampling="systematic", k=0.1, stratified_col=None)
print(sysp.count())


# In[ ]:




