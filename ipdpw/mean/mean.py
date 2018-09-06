#  -*- coding: utf-8 -*-
import sys
import os
from pandas import DataFrame   #DataFrame通常来装二维的表格
import pandas as pd            #pandas是流行的做数据分析的包
import numpy as np
 
#建立字典，键和值都从文件里读出来。键是nam，age……，值是lili，jim……
data = pd.read_table('extract.txt', sep='\t',names=["pos","read","SlnIndex","ipd","pw"],low_memory=False,header=None,skiprows=[0])
df = DataFrame(data,columns=["pos","read","SlnIndex","ipd","pw"])
print(df)
df1 = []
#pos = df.pos.unique()
l=[]
#print(pos)
for p in range(13794):
    a = df[df['pos']==p+1]['ipd'].mean()
    b = df[df['pos']==p+1]['pw'].mean()
    l=[a,b]
    df1.append(l)
df2=DataFrame(df1)
print(df2)
df2.to_csv('mean.txt',index=False,header=["ipd_mean","pw_mean"],sep='\t')
df2.to_csv('mean.xlsx',index=False,header=["ipd_mean","pw_mean"],sep='\t')

