#  -*- coding: utf-8 -*-
import sys
import os
from pandas import DataFrame   #DataFrame通常来装二维的表格
import pandas as pd            #pandas是流行的做数据分析的包
import numpy as np
 
#建立字典，键和值都从文件里读出来。键是nam，age……，值是lili，jim……
data = pd.read_table('6mA_ChuanHe25_holes_concat_neg.txt', sep=' ',names=None,low_memory=False,header=None)
df = DataFrame(data,columns=None)
df=df[df[0]==6]
print(df)
df.to_csv('M_neg.txt',index=False,header=None,sep=' ')
