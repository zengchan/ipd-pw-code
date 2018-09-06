#coding:utf8
import sys
import os
from pandas import DataFrame   #DataFrame通常来装二维的表格
import pandas as pd            #pandas是流行的做数据分析的包
import numpy as np
 
#建立字典，键和值都从文件里读出来。键是nam，age……，值是lili，jim……
dna_dir = './dna'
data = pd.read_table('sort.txt', sep='\t',names=None,low_memory=False,header=None)
df = DataFrame(data)
df=df.drop([0])
df=df.reset_index(drop=True)
dfe = pd.DataFrame()
print(df)

def read_fasta():
    with open(os.path.join(dna_dir,'Caenorhabditis_elegans.ce10.dna.chromosome.M.fa')) as fa:
         dna = fa.read().replace('\n', '').upper()[5:] #not consist the annotation>chr1
    return dna
dna_seq=[]
dna_seq=read_fasta()

for p in range(len(dna_seq)):
    df1=df[(True ^ (df[1] != dna_seq[p])) & (True ^ (df[0]!=p+1))]
    dfe=pd.concat([dfe,df1])
print(dfe)
dfe.to_csv('extract.txt',index=False,header=["pos","read","SlnIndex","ipd","pw"],sep='\t')
dfe.to_csv('extract.xlsx',index=False,header=["pos","read","SlnIndex","ipd","pw"],sep='\t')
