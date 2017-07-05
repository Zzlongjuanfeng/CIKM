#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""writing recognition in tensorflow"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import data_process
import os
import time
import numpy

def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())


def inference(images, keep_prob, featureMap1 = 64, featureMap2 = 128, featureMap3 = 256, featureMap4 = 512):
    """
        Build the neuron network model.
    """
    parameters = []

    # the first convolution layer
    with tf.name_scope('conv1'):
        # convolution kernel
        kernel = tf.Variable( tf.truncated_normal(shape=[7, 7, 15, featureMap1], stddev=0.05),
                              name='weights')
        biases = tf.Variable( tf.constant(value=0.0, shape=[featureMap1]),
                              name='biases')
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.softplus(conv + biases)
        print_activations(conv1)
        parameters += [kernel, biases]

    # first pool layer
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    print_activations(pool1)

    # the second convolution layer
    with tf.name_scope('conv2'):
        kernel = tf.Variable( tf.truncated_normal(shape=[5, 5, featureMap1, featureMap2], stddev=0.05),
                              name='weights')
        biases = tf.Variable( tf.constant(value=0.0, shape=[featureMap2]),
                              name='biases')
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.softplus(conv + biases)
        print_activations(conv2)
        parameters += [kernel, biases]

    # second pool layer
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')
    print_activations(pool2)

    # the third convolution layer
    with tf.name_scope('conv3'):
        kernel = tf.Variable( tf.truncated_normal(shape=[3, 3, featureMap2, featureMap3], stddev=0.05),
                              name='weights')
        biases = tf.Variable( tf.constant(value=0.0, shape=[featureMap3]),
                              name='biases')
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.softplus(conv + biases)
        print_activations(conv3)
        parameters += [kernel, biases]

    # the 4th convolution layer
    with tf.name_scope('conv4'):
        kernel = tf.Variable( tf.truncated_normal(shape=[3, 3, featureMap3, featureMap4], stddev=0.05),
                              name='weights')
        biases = tf.Variable( tf.constant(value=0.0, shape=[featureMap4]),
                              name='biases')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.softplus(conv + biases)
        print_activations(conv4)
        parameters += [kernel, biases]

    # full connect layer
    shp = conv4.get_shape()
    dim = shp[1].value * shp[2].value * shp[3].value
    conv4_flat = tf.reshape(conv4, [-1,dim])
    fc1 = slim.fully_connected(inputs=conv4_flat, num_outputs=512, activation_fn=tf.nn.relu, scope='fc1')
    fc1_drop = tf.nn.dropout(fc1, keep_prob)  # dropout layer

    fc2 = slim.fully_connected(inputs=fc1_drop, num_outputs=1, activation_fn=None, scope='fc2')
    return tf.reshape(fc2, [-1])


def net(image, i_th):
    name = 'net_common' + str(i_th)
    with tf.variable_scope(name):
        conv1 = slim.conv2d(inputs = image, num_outputs = 96, kernel_size= [7, 7],
                            stride = 2, scope = 'conv1')

        pool1 = slim.max_pool2d(inputs=conv1, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')

        conv2 = slim.conv2d(inputs = pool1, num_outputs = 256, kernel_size= [5, 5],
                            stride = 1, scope = 'conv2')

        pool2 = slim.max_pool2d(inputs=conv2, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool2')

        conv3 = slim.conv2d(inputs = pool2, num_outputs = 384, kernel_size= [3, 3],
                            stride = 1, scope = 'conv3')

        conv4 = slim.conv2d(inputs = conv3, num_outputs = 384, kernel_size= [3, 3],
                            stride = 1, scope = 'conv4')

        conv5 = slim.conv2d(inputs = conv4, num_outputs = 256, kernel_size= [3, 3],
                            stride = 1, scope = 'conv5')

        pool5 = slim.max_pool2d(inputs=conv5, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool5')

        # full connect layer
        shp = pool5.get_shape()
        dim = shp[1].value * shp[2].value * shp[3].value
        pool5_flat = tf.reshape(pool5, [-1,dim])
        fc1 = slim.fully_connected(inputs=pool5_flat, num_outputs=1024,activation_fn=tf.nn.relu, scope='fc1')
        fc1_drop = tf.nn.dropout(fc1, keep_prob)  # dropout layer

        fc2 = slim.fully_connected(inputs=fc1_drop, num_outputs=1, activation_fn=tf.nn.relu, scope='fc2')
        if i_th == 0 :
            print_activations(conv1)
            print_activations(pool1)
            print_activations(conv2)
            print_activations(pool2)
            print_activations(conv3)
            print_activations(conv4)
            print_activations(conv5)
            print_activations(pool5)
            print_activations(fc1_drop)

    return fc2



def learning_decay():
    # learning rate changes during training step
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(1e-3, global_step, 45, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss, global_step=global_step)

def adjustment(datafile):
    line_offset = []
    count = 0
    with open(datafile, 'r') as file:
        for line in file:
            line_offset.append(float(line.strip()) - 1)
            count+=1
            print(count)
    outName = os.path.splitext(datafile)[0] + '_new.csv'
    # print(outName)
    with open(outName, 'w') as out_f:
        for i in line_offset:
            out_f.write(str(i)+'\n')

if __name__ == "__main__":

    data_file = '/home/zxf/PycharmProjects/CIKM/CNN_new/out/1498834676-model-9000B.csv'
    adjustment(data_file)