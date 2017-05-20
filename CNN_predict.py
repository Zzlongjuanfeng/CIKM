#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""writing recognition in tensorflow"""
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()    # initial session
x = tf.placeholder(tf.float32, [None, 101 * 101 * 60])    # input data
y_ = tf.placeholder(tf.float32, [None, 1])    # label

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

# the second convolution layer
W_conv3 = weight_variable([5, 5, 128, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# fc
shp = h_pool3.get_shape()
dim = shp[1].value() * shp[2].value() * shp[3].value()
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
loss = tf.abs(tf.sub(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())



#training
for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i%500 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


print("test accuracy %g"%accuracy.eval(feed_dict={
    x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))
