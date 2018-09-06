#  -*- coding: utf-8 -*-
import tensorflow as tf
import data_helper as dhrt
import os
import numpy as np
import random
tf.reset_default_graph()


# Hyper Parameters
learning_rate = 0.01    # 学习率
n_steps = 11            # LSTM 展开步数（时序持续长度）
n_inputs = 1           # 输入节点数
n_hiddens = 64         # 隐层节点数
n_layers = 1            # LSTM layer 层数
n_classes = 2          # 输出节点数（分类数目）
epoch = 100
batchsize = 128
batchsize1 = 64

data_dir = "./data"
x_rt_train, y_rt_train = dhrt.load_data_and_labels(os.path.join(data_dir, 'train_positive.txt'),
                                                           os.path.join(data_dir, 'train_negative.txt'))

#l_x_rt_train = len(y_rt_train)


index= [i for i in range(len(y_rt_train))]
# print(type(index))
random.shuffle(index)
index=np.array(index)
x_rt_train_shuffled=list(np.array(x_rt_train)[index])
y_rt_train_shuffled=list(np.array(y_rt_train)[index])
x_rt_train_shuffled1=list()
print(x_rt_train_shuffled)
#x_rt_train_shuffled=[[i for i in ii.split(',')] for ii in x_rt_train_shuffled]
for row in range(len(x_rt_train_shuffled)):
    x = x_rt_train[row].split(' ')
    x = list(map(eval, x))
    x_rt_train_shuffled1.append(x)
#print(len(x_rt_train_shuffled1))
x_rt_test, y_rt_test = dhrt.load_data_and_labels(os.path.join(data_dir, 'test_positive.txt'),
                                                           os.path.join(data_dir, 'test_negative.txt'))

index1= [i for i in range(len(y_rt_test))]
# print(type(index))
random.shuffle(index1)
index1=np.array(index1)
x_rt_test_shuffled=list(np.array(x_rt_test)[index1])
y_rt_test_shuffled=list(np.array(y_rt_test)[index1])
x_rt_test_shuffled1=list()
print(x_rt_test_shuffled)
#print(x_rt_train_shuffled1)
#print(x_rt_train_shuffled)
#x_rt_train_shuffled=[[i for i in ii.split(',')] for ii in x_rt_train_shuffled]
for row in range(len(x_rt_test_shuffled)):
    x = x_rt_test[row].split(' ')
    x = list(map(eval, x))
    x_rt_test_shuffled1.append(x)
#print(x_rt_test_shuffled1)
# tensor placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_steps * n_inputs], name='x_input')     # 输入
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')               # 输出
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')           # 保持多少不被 dropout
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')       # 批大小

# weights and biases
with tf.name_scope('weights'):
    Weights = tf.Variable(tf.truncated_normal([n_hiddens, n_classes],stddev=0.1), dtype=tf.float32, name='W')
    tf.summary.histogram('output_layer_weights', Weights)
with tf.name_scope('biases'):
    biases = tf.Variable(tf.random_normal([n_classes]), name='b')
    tf.summary.histogram('output_layer_biases', biases)



def RNN_LSTM(x, Weights, biases):
    # RNN 输入 reshape
    x = tf.reshape(x, [-1, n_steps, n_inputs])
    # 定义 LSTM cell
    # cell 中的 dropout
    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope('lstm_dropout'):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # attn_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # 实现多层 LSTM
    # [attn_cell() for _ in range(n_layers)]
    enc_cells = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
    with tf.name_scope('lstm_cells_layers'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)
    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn 运行网络
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
    #return outputs
    # 输出
    #return tf.matmul(outputs[:,-1,:], Weights) + biases
    return tf.nn.softmax(tf.matmul(outputs[:,-1,:], Weights) + biases)

with tf.name_scope('output_layer'):
    pred = RNN_LSTM(x, Weights, biases)
    tf.summary.histogram('outputs', pred)
# cost
with tf.name_scope('loss'):
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices=[1]))
    tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1))[1]
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
def train_lstm():
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter("train", sess.graph)
        # training
        step = 1
        for i in range(500):
            batch_train = dhrt.batch_iter(x_rt_train_shuffled1, y_rt_train_shuffled, 128)
            for x_batch_train, y_batch_train in batch_train:
                #print(y_batch_train)
                sess.run(train_op, feed_dict={x: x_batch_train, y: y_batch_train, keep_prob: 0.5, batch_size: batchsize})
                if (i + 1) % 50 == 0:
                    train_result = sess.run(merged,
                                            feed_dict={x: x_batch_train, y: y_batch_train, keep_prob: 1.0, batch_size: batchsize})
                    loss = sess.run(cost, feed_dict={x:x_batch_train, y:y_batch_train, keep_prob:1.0, batch_size:batchsize})
                    acc = sess.run(accuracy, feed_dict={x:x_batch_train, y:y_batch_train, keep_prob:1.0, batch_size:batchsize})
                    print('Iter: %d' % ((i+1) * batchsize), '| train loss: %.6f' % loss, '| train accuracy: %.6f' % acc)
                    train_writer.add_summary(train_result, i + 1)
                    print("保存模型：", saver.save(sess, 'ckpt/prediction.model'))

                # for x_batch_test, y_batch_test in batch_test:
                #     y_batch_test=y_batch_test.reshape(-1, 2)
                #     print(y_batch_test)
                #     test_result = sess.run(merged,
                #                        feed_dict={x: x_batch_test, y: y_batch_test, keep_prob: 1.0,
                #                                   batch_size: batchsize1})
                #     test_writer.add_summary(test_result, i + 1)
                #     print(test_result)
        print("Optimization Finished!")
#train_lstm()
def prediction():
    with tf.Session() as sess:
        #参数恢复
        sess.run(init)
        test_writer = tf.summary.FileWriter("test", sess.graph)
        module_file = tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, module_file)
        batch_test = dhrt.batch_iter(x_rt_test_shuffled1, y_rt_test_shuffled, 64)
        for x_batch_test, y_batch_test in batch_test:
            #print(len(x_batch_test))
            y_batch_test = y_batch_test.reshape(-1, 2)
            #print(y_batch_test)
                # prediction
            print("Testing Accuracy:",
                      sess.run(accuracy, feed_dict={x: x_batch_test, y: y_batch_test, keep_prob: 1.0, batch_size: batchsize1}))
prediction()




