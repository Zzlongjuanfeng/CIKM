#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""writing recognition in tensorflow"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import data_process
import os
import time
import numpy as np
import some_code
import numpy
import csv
test_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testA/testA.txt"
dir_load = "/home/zxf/PycharmProjects/CIKM/CNN_new/run/1497984104/checkpoints/model-15000"
dir_out = "/home/zxf/PycharmProjects/CIKM/CNN_new/out"
test_batchSize=100
# data_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train.txt"
# train_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train_file.txt"
# veri_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/veri_file.txt"

# data_dir = os.path.abspath(os.path.join(os.path.curdir, "CIKM_data", "CIKM2017_train"))
# veri_file = os.path.join(data_dir, "veri_file.txt")
# train_file = os.path.join(data_dir, "train_file.txt")
# data_file = os.path.join(data_dir, "train.txt")

# veri_size = 10
veri_size = batch_size = 100

def print_activations(t):
    print(t.op.name, t.get_shape().as_list())

def net(image, i):
    name = 'net_common' + str(i)
    with tf.variable_scope(name):
        conv1 = slim.conv2d(inputs = image, num_outputs = 64, kernel_size= [7, 7],
                            stride = 3, scope = 'conv1')

        pool1 = slim.max_pool2d(inputs=conv1, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')

        conv2 = slim.conv2d(inputs = pool1, num_outputs = 128, kernel_size= [5, 5],
                            stride = 1,reuse=True, scope = 'conv2')

        pool2 = slim.max_pool2d(inputs=conv2, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool2')

        conv3 = slim.conv2d(inputs = pool2, num_outputs = 256, kernel_size= [5, 5],
                            stride = 1, scope = 'conv3')

        conv4 = slim.conv2d(inputs = conv3, num_outputs = 512, kernel_size= [5, 5],
                            stride = 1, scope = 'conv4')
        # full connect layer
        shp = conv4.get_shape()
        dim = shp[1].value * shp[2].value * shp[3].value
        conv4_flat = tf.reshape(conv4, [-1,dim])
        fc1 = slim.fully_connected(inputs=conv4_flat, num_outputs=512,activation_fn=tf.nn.relu, scope='fc1')
        fc1_drop = tf.nn.dropout(fc1, keep_prob)  # dropout layer

        print_activations(conv1)
        print_activations(pool1)
        print_activations(conv2)
        print_activations(pool2)
        print_activations(conv3)
        print_activations(conv4)
        print_activations(fc1_drop)

    # with tf.variable_scope('fc2'):
    #     fc2 = slim.fully_connected(inputs=cell_output, num_outputs=1, activation_fn=None, scope='fc2')
    # return tf.reshape(fc2, [-1])
    return fc1_drop

#Graph
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 15* 4 * 101 * 101])    # input data
    y_ = tf.placeholder(tf.float32, [None,])    # label
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 15, 4, 101, 101])
    x_image = tf.transpose(x_image, [0,3,4,1,2])
    x_image_h1 = x_image[:, :, :, :, 1]
    x_image_h0 = x_image[:, :, :, :, 0]
    x_image_h2 = x_image[:, :, :, :, 2]
    x_image_h3 = x_image[:, :, :, :, 3]

    # x_image_h1 = tf.reshape(x_image_h1, [-1, 101, 101 ,1])
    # x_image_h1 = tf.transpose(x_image_h1, [0, 2, 3, 1])

print(x_image_h1.shape)
fc1_h0 = net(x_image_h0, 0)
fc1_h1 = net(x_image_h1, 1)
fc1_h2 = net(x_image_h2, 2)
fc1_h3 = net(x_image_h3, 3)

fc1_all = tf.concat([fc1_h0, fc1_h1, fc1_h2, fc1_h3], axis=1)
print(fc1_all.shape)
# fc2 = slim.fully_connected(inputs=fc1_all, num_outputs=20, activation_fn=tf.nn.relu, scope='fc2')
y_conv = slim.fully_connected(inputs=fc1_all, num_outputs=1, activation_fn=None, scope='fc2')
y_conv = tf.reshape(y_conv, [-1])

# loss function
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_conv, y_))))

# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-4, global_step, 95, 0.8, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(loss, global_step=global_step)

# gpu configuration
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.visible_device_list = '0'
# gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


with tf.Session(config=tf_config) as sess:

    saver = tf.train.Saver(max_to_keep = None)
    # Initialize parameters
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, dir_load)

    #Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", timestamp))
    print("Writing to {}\n".format(out_dir))

    #testing
    test_batches = data_process.batch_iter1(test_file, test_batchSize)
    testOutAll = []
    count = 0
    for batch in test_batches:
        count += 1
        testLoss_out, test_out= sess.run([loss, y_conv], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, test loss %g" % (count * test_batchSize, testLoss_out))
        testOut_flat = test_out.reshape(-1)
        testOutAll = np.concatenate([testOutAll, testOut_flat])

        # print("the predict:", test_out)
        if (count * test_batchSize) % 2000 == 0:
            testOutAll_column = np.column_stack(testOutAll)
            outName = os.path.split(dir_load)
            out_name = os.path.split( os.path.split(outName[0])[0] )[1] + '-' + os.path.splitext(outName[1])[0] + '.csv'
            out_path = os.path.join(dir_out, out_name)
            print("Saving evaluation to {0}".format(out_path))
            with open(out_path, 'w') as f:
                csv.writer(f, delimiter = '\n').writerows(testOutAll_column)




