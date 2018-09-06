#coding:utf8
import sys
from pandas import DataFrame   #DataFrame通常来装二维的表格
import pandas as pd            #pandas是流行的做数据分析的包
import numpy as np
 
#建立字典，键和值都从文件里读出来。键是nam，age……，值是lili，jim……

data = pd.read_table('chrI_ipd_pw_HeChuan36c_ccssubsort.txt', sep='\t',names=["pos","read","SlnIndex","ipd","pw"],usecols=[0,1,3,5,6])
df = DataFrame(data,columns=["pos","read","SlnIndex","ipd","pw"])
#print(df)
#把DataFrame输出到一个表，不要行名字和列名字
#df.to_csv('extract.xlsx',index=False,header=["pos","read","SlnIndex","ipd","pw"],sep='\t')
#df.to_csv('extract.txt',index=False,header=["pos","read","SlnIndex","ipd","pw"],sep='\t')
df1=df.sort_values(by=['pos','read'])
#print(df1)
df1.to_csv('sort.xlsx',index=False,header=["pos","read","SlnIndex","ipd","pw"],sep='\t')
df1.to_csv('sort.txt',index=False,header=["pos","read","SlnIndex","ipd","pw"],sep='\t')


