# -*- coding:utf-8 -*-
import numpy as np
import re
import math
import itertools
from collections import Counter
import random

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    return string.strip().lower()


def load_data_and_labels(pos_file,neg_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(pos_file, "r").readlines())
    positive_examples = [s.strip('\n') for s in positive_examples]#用空格分割每个碱基
    negative_examples = list(open(neg_file, "r").readlines())
    negative_examples = [s.strip('\n') for s in negative_examples]
    #print(negative_examples)
    # Split by words
    x_text = positive_examples + negative_examples
    #x_text = [clean_str(sent) for sent in x_text]
    #print(x_text)
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(x, y, batch_size):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(y)
    num_batches = math.ceil(data_size / batch_size) #用于向上取整数计算,返回的是大于或等于函数参数的数值
    #
    index = [i for i in range(data_size)]
    # print(type(index))
    random.shuffle(index)
    index = np.array(index)
    x_shuffle = np.array(x)[index]
    y_shuffle = np.array(y)[index]
    # #print(x_shuffle)
    # indices = np.random.permutation(np.arange(data_size))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    #print(x_shuffle)

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_size)
        yield x_shuffle[start_index:end_index], y_shuffle[start_index:end_index]
        # print(y_shuffle)
        # print(len(x_shuffle))


