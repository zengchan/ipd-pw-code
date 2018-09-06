#  -*- coding: utf-8 -*-
import sys
import os
import math
import random 
from pandas import DataFrame   #DataFrame通常来装二维的表格
import pandas as pd            #pandas是流行的做数据分析的包
import numpy as np
 
sys.setrecursionlimit(1000000)
positive_dir = './M_pos.txt'
negative_dir = './M_neg.txt'

df1=[]

def load_txt(file_dir):
    data = []
    with open(file_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.split(' '))#Note sometimes split(' ')
    return data
    # print(type(data[1][1]))


def list_chrom(data):
    column0 = [x[0] for x in data]
    chroms = sorted(set(column0), key=column0.index)
    print('the chromosomes:', chroms)
    return chroms

def read_ipdpw():
    data1 = []
    with open('pw.txt', 'r') as f:
         lines = f.readlines()  
         for line in lines:
             line=line.strip('\n') 
             data1.append(float(line))#Note sometimes split(' ')
    return data1 

def write_fasta(kmers, filename):
    with open(os.path.join('data',filename), 'w') as fh:
        for i, kmer in enumerate(kmers):
            #print('>%d' % i, file=fh)
            print(kmer, file=fh)

def positive():
    print('loading txt file which consisting chr,position,label')
    data_positive = load_txt(positive_dir)  #shape:list
    num_positive = len(data_positive)
    random.shuffle(data_positive)
    print('the number of positive samples: ',num_positive)
    val_positive_num = math.ceil(num_positive * 0.3)
    val_positive = data_positive[:val_positive_num]
    print('the number of validation positive samples: ', val_positive_num)
    train_positive = data_positive[val_positive_num:]
    print('the number of train positive samples: ', len(train_positive))
    data_dict = {'train_positive':train_positive ,'val_positive':val_positive}
    for key,value in data_dict.items():
#       print('list the chromosomes' + key)
        chroms_pos = list_chrom(value)
        dna_seqs = []
        for chrom in chroms_pos:
            dna_seq = read_ipdpw()
            dna_seqs.append(dna_seq)
        subdnas = pd.DataFrame()
        for x in value:
            p = chroms_pos.index(x[0])
            position = int(x[1]) - 1
            subdna = dna_seqs[p][(position - 5):(position + 5 + 1)]
            subdna1=DataFrame(subdna)
            subdna1=subdna1.T
            subdnas=pd.concat([subdnas,subdna1])
        subdnas = subdnas.round(4)
        subdnas.to_csv(key+'.txt',index=False,header=None,sep=' ')
        #subdnas.to_csv(key+'.xlsx',index=False,header=None,sep=' ')
def negative():
    print('loading txt file which consisting chr,position,label')
    data_negative = load_txt(negative_dir)  # shape:list
    num_negative = len(data_negative)
    random.shuffle(data_negative)
    print('the number of negative samples: ', num_negative)
    val_negative_num = math.ceil(num_negative * 0.3)
    val_negative = data_negative[:val_negative_num]
    print('the number of validation negative samples: ', val_negative_num)
    train_negative = data_negative[val_negative_num:]
    print('the number of train negative samples: ', len(train_negative))
    data_dict = {'train_negative': train_negative, 'val_negative': val_negative}
    for key, value in data_dict.items():
#       print('list the chromosomes'+ key)
        chroms_pos = list_chrom(value)
        dna_seqs = []
        for chrom in chroms_pos:
            dna_seq = read_ipdpw()
            dna_seqs.append(dna_seq)
        # print(dna_seqs[-1][-20:])
        #subdnas = []
        subdnas = pd.DataFrame()
        #array1=np.array(subdnas)
        for x in value:
            #print(array1)
            p = chroms_pos.index(x[0])
            position = int(x[1]) - 1
            subdna = dna_seqs[p][(position - 5):(position + 5 + 1)]
            subdna1=DataFrame(subdna)
            subdna1=subdna1.T
            subdnas=pd.concat([subdnas,subdna1])
        subdnas = subdnas.round(4)
        print(subdnas)
        subdnas.to_csv(key+'.txt',index=False,header=None,sep=' ')
        #subdnas.to_csv(key+'.xlsx',index=False,header=None,sep=' ')
        #print(array1.shape[1])
        #np.savetxt(key+'.txt', array1,fmt='%s' *int((array1.size/len(array1))),newline='\n')

if __name__ == '__main__':
    if sys.argv[1] not in ['positive','negative']:
        raise ValueError("""usage: python pos_to_fasta_split.py [positive / negative]""")
    if sys.argv[1] == 'positive':
        positive()
    else:
        negative()
















