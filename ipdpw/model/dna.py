import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn


def seq_pre_processor(x, sess, max_document_length, vocabulary_length):
    # max_len = x.shape[1]
    # num_regions = max_len-region_size+1
    # N = vocabulary_length*region_size
    # x_out = np.zeros((N,num_regions))

    pre_x = tf.placeholder(tf.int32,
                           [None, max_document_length]
                           )
    feed_dict = {pre_x: x}
    pre_processor = tf.one_hot(pre_x, vocabulary_length)
    p = sess.run([pre_processor], feed_dict)
    p = p[0]   #2-dim
    x_out = p.reshape(x.shape[0], -1, 1, 1) #4-dim

    return x_out

class CNNParam(object):
    num_classes = 2
    num_filters = [128, 256]
    region_size = 11
    evaluate_every = 2000
    dropout_param = 0.5


class seqCNN(object):
    """
    This CNN is an atempt to recreate the CNN described in this paper:
    Effective Use of Word Order for Text Categorization with Convolutional Neural Networks
    Rie Johnson, Tong Zhang
    """

    def __init__(self, num_classes, num_filters, num_pooled, region_size, max_sentence_length,
                 l2_reg_lambda, filter_lengths=3):
        # input layers and params
        #filter_length = vocabulary_length * region_size
        #sentence_length = max_sentence_length * vocabulary_length
        self.x_input = tf.placeholder(tf.float32,[None, max_sentence_length, 4], name="x_input")
        self.y_input = tf.placeholder(tf.float32,[None, num_classes], name="y_input")

        self.dropout_param = tf.placeholder(tf.float32, name="dropout_param")

        cnn_filter_shape = [region_size, 4, num_filters[0]]
        initializer1w = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
        initializer1b = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
        W_CN = tf.get_variable(name="W_CN", shape=cnn_filter_shape, dtype=tf.float32, initializer=initializer1w)
        b_CN = tf.get_variable(name="b_CN", shape=[num_filters[0]], dtype=tf.float32, initializer=initializer1b)
        W_CN = tf.Variable(tf.truncated_normal(cnn_filter_shape, stddev=0.1), name="W_CN")
        b_CN = tf.Variable(tf.truncated_normal([num_filters[0]], stddev=0.1), name="b_CN")

        cnn_filter2_shape = [filter_lengths,128,num_filters[1]]

        initializer2w = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
        initializer2b = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
        W_CN2 = tf.get_variable(name="W_CN2", shape=cnn_filter2_shape, dtype=tf.float32, initializer=initializer2w)
        b_CN2 = tf.get_variable(name="b_CN2", shape=[num_filters[1]], dtype=tf.float32, initializer=initializer2b)
        W_CN2 = tf.Variable(tf.truncated_normal(cnn_filter2_shape, stddev=0.1), name="W_CN2")
        b_CN2 = tf.Variable(tf.truncated_normal([num_filters[1]], stddev=0.1), name="b_CN2")

        # conv-relu-pool layer
        conv1 = tf.nn.conv1d(
            self.x_input,
            filters=W_CN,
            stride=1,
            padding="VALID",
            #name="name"
        )
        self.relu1 = tf.nn.sigmoid(
            tf.nn.bias_add(conv1, b_CN),
            name="sigmoid1"
        )
        print(self.relu1.shape)
        conv2 = tf.nn.conv1d(
            self.relu1,
            filters=W_CN2,
            stride=1,
            padding="SAME",
            name="conv2"
        )

        relu2 = tf.nn.sigmoid(
            tf.nn.bias_add(conv2, b_CN2),
            name="sigmoid2"
        )
        newrelu2=tf.expand_dims(relu2,axis=2)
        pool_stride = [1, int((max_sentence_length - region_size + 1) / num_pooled), 1,1]
        pool = tf.nn.avg_pool(
            newrelu2,
            ksize=pool_stride,
            strides=pool_stride,
            padding="VALID",
            name="pool"
        )
        print(pool)
        # dropout
        drop = tf.nn.dropout(
            pool,
            self.dropout_param
        )

        # response normalization
        normalized = tf.nn.local_response_normalization(drop)
        return normalized
       
