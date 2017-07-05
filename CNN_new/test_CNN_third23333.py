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
# test_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testA/testA.txt"
test_fileB = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testB/testB.txt"
dir_load = "/home/zxf/PycharmProjects/CIKM/CNN_new/run/1498656455/checkpoints/model-12000"
dir_out = "/home/zxf/PycharmProjects/CIKM/CNN_new/out"
model_file = "/home/zxf/PycharmProjects/CIKM/CNN_new/run/1498656455/checkpoints/model-12000.meta"
test_batchSize=100


# veri_size = 10
veri_size = batch_size = 100

def print_activations(t):
    print(t.op.name, t.get_shape().as_list())

# with tf.name_scope('input'):
#     x = tf.placeholder(tf.float32, [None, 15* 4 * 101 * 101])    # input data
#     y_ = tf.placeholder(tf.float32, [None,])    # label
#     keep_prob = tf.placeholder(tf.float32)

# gPu configuration
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.visible_device_list = '1'
# gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf_config) as sess:

    saver = tf.train.import_meta_graph(model_file)
    # saver = tf.train.Saver(max_to_keep = None)
    # Initialize parameters
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, dir_load)

    x = tf.get_default_graph().get_tensor_by_name('input/Placeholder:0')
    y_ = tf.get_default_graph().get_tensor_by_name('input/Placeholder_1:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('input/Placeholder_2:0')

    y_conv = tf.get_default_graph().get_tensor_by_name('fc3/BiasAdd:0')
    loss =  tf.get_default_graph().get_tensor_by_name('loss:0')

    collection = tf.get_default_graph().get_all_collection_keys()

    #Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", timestamp))
    print("Writing to {}\n".format(out_dir))

    #testing
    test_batches = data_process.batch_iter1(test_fileB, test_batchSize)
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
            out_name = os.path.split( os.path.split(outName[0])[0] )[1] + '-' + os.path.splitext(outName[1])[0] + 'B.csv'
            out_path = os.path.join(dir_out, out_name)
            print("Saving evaluation to {0}".format(out_path))
            with open(out_path, 'w') as f:
                csv.writer(f, delimiter = '\n').writerows(testOutAll_column)




