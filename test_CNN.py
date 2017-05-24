#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""writing recognition in tensorflow"""
import tensorflow as tf
import data_process
import csv
import numpy as np
import os


batch_size = 50
data_size = 10000   # the number of training data
data_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train.txt"

test_batchSize = 100
test_size = 2000
test_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testA/testA.txt"

dir_load = "/home/zxf/PycharmProjects/CIKM/run5/epochs2_10000.ckpt"
dir_out = "/home/zxf/PycharmProjects/CIKM/out"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#Graph
x = tf.placeholder(tf.float32, [None, 101 * 101 * 60])    # input data
y_ = tf.placeholder(tf.float32, [None,])    # label

# the first convolution layer
x_image = tf.reshape(x, [-1, 101, 101, 60])
W_conv1 = weight_variable([5, 5, 60, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# the second convolution layer
W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# the third convolution layer
W_conv3 = weight_variable([5, 5, 128, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# fc
shp = h_pool3.get_shape()
dim = shp[1].value * shp[2].value * shp[3].value
W_fc1 = weight_variable([dim, 2048])
b_fc1 = bias_variable([2048])
h_pool3_flat = tf.reshape(h_pool3, [-1, dim])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([2048, 1])
b_fc2 = bias_variable([1])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss function
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_conv, y_))))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# gpu configuration
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = None)

with tf.Session(config=tf_config) as sess:
    # sess.run(init)
    saver.restore(sess, dir_load)

    #testing
    test_batches = data_process.batch_iter1(test_file, test_batchSize)
    testOutAll = []
    for batch in test_batches:
        testLoss_out, test_out = sess.run([loss, y_conv], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, test loss %g" % (batch[2], testLoss_out))
        testOut_flat = test_out.reshape(-1)
        # print(test_out.shape)
        testOutAll = np.concatenate([testOutAll, testOut_flat])

        # print("the predict:", test_out)
        if batch[2] % 2000 == 0:
            testOutAll_column = np.column_stack(testOutAll)
            out_path = os.path.join(dir_out, "prediction_60000_slow.csv")
            print("Saving evaluation to {0}".format(out_path))
            with open(out_path, 'w') as f:
                csv.writer(f, delimiter = '\n').writerows(testOutAll_column)




