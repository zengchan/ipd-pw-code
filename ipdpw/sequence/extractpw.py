#  -*- coding: utf-8 -*-
import sys
import os
from pandas import DataFrame   #DataFrame通常来装二维的表格
import pandas as pd            #pandas是流行的做数据分析的包
import numpy as np
df1=[]
data = pd.read_table('mean.txt', sep='\t',names=None,low_memory=False,header=None,skiprows=[0])
df = DataFrame(data,columns=None)
df1=df[[1]] 
print(df1)
df1.to_csv('pw.txt',index=False,header=None,sep=' ')
print(np.array(df1).shape)
