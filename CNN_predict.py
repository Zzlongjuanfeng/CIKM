#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""writing recognition in tensorflow"""
import tensorflow as tf
import data_process
import os
import time

batch_size = 50
data_size = 10000   # the number of training data
data_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train.txt"

test_batchSize = 400
test_size = 2000
test_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_testA/testA.txt"

data_dir = os.path.abspath(os.path.join(os.path.curdir, "CIKM_data", "CIKM2017_train"))
veri_file = os.path.join(data_dir, "veri_file.txt")
train_file = os.path.join(data_dir, "train_file.txt")
veri_size = 100

dir_model = "/home/zxf/PycharmProjects/CIKM/run/run_new_1"
dir_load = "/home/zxf/PycharmProjects/CIKM/run/1495899128/checkpoints/model-24000"

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
W_conv3 = weight_variable([5, 5, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# fc1
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
train_step = tf.train.AdamOptimizer(5e-4).minimize(loss)

# gpu configuration
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf_config) as sess:
    #Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", timestamp))
    print("Writing to {}\n".format(out_dir))

    #summary for loss
    loss_summary = tf.summary.scalar("loss", loss)
    # merged = tf.merge_all_summaries()
    grad_summaries = []
    grad_summaries.append(loss_summary)
    merged = tf.summary.merge(grad_summaries)

    #train summary
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    #dev summary
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(max_to_keep = None)

    # Initialize parameters
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, dir_load)

    #training
    count = 0
    for i in range(5):
        train_batches = data_process.batch_iter1(train_file, batch_size)
        for batch in train_batches:
            count += 1
            # vari_x.extend(batch[2])
            # vari_y.extend(batch[3])
            #training with a batch
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            #training loss
            if (count * batch_size) % 200 == 0:
                train_summary, train_loss_out= sess.run([merged, loss], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
                print("epochs %d, step %d, training loss %g" % (i, count * batch_size, train_loss_out))
                train_summary_writer.add_summary(train_summary, count)

            #varification
            if (count * batch_size) % 3000 == 0:
                veri_batches = data_process.batch_iter1(veri_file, veri_size)
                # veri_batch = next(veri_batches)
                veri_loss_mean = 0
                veri_count = 0
                for veri_batch in veri_batches:
                    vari_summary, veri_loss_out= sess.run([merged, loss], feed_dict={x: veri_batch[0], y_: veri_batch[1], keep_prob: 1})
                    veri_loss_mean += veri_loss_out
                    veri_count += 1
                    # print(veri_count)
                print('='*10)
                print("*****epochs %d, step %d, varification loss %g" % (i, count * batch_size, veri_loss_mean / veri_count))
                print('='*10)
                dev_summary_writer.add_summary(vari_summary, count)

            #save model
            if (count * batch_size) % 3000 == 0:
                # if not os.path.exists(dir_model):
                path = saver.save(sess, checkpoint_prefix, global_step = count * batch_size)
                print("Saved model checkpoint to {}\n".format(path))
                #     os.makedirs(dir_model)
                # saver.save(sess, (dir_model + "/epochs%d_%d.ckpt") % (i, count * batch_size))





