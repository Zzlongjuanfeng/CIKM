#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""writing recognition in tensorflow"""
"""
the net has four inputs and lstm
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import data_process
import os
import time
import some_code
import numpy as np
import math

data_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train.txt"
train_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/train_file.txt"
veri_file = "/home/zxf/PycharmProjects/CIKM/CIKM_data/CIKM2017_train/veri_file.txt"

# data_dir = os.path.abspath(os.path.join(os.path.curdir, "CIKM_data", "CIKM2017_train"))
# veri_file = os.path.join(data_dir, "veri_file.txt")
# train_file = os.path.join(data_dir, "train_file.txt")
# data_file = os.path.join(data_dir, "train.txt")

# veri_size = 10
veri_size = 100
batch_size = 100
dir_load = "/home/zxf/PycharmProjects/CIKM/run/1496093293/checkpoints/model-18000"

def print_activations(t):
    print(t.op.name, t.get_shape().as_list())

def net(image, i, keep_prob):
    name = 'net_common' + str(i)
    with tf.variable_scope(name, reuse=True):
        conv1 = slim.conv2d(inputs = image, num_outputs = 32, kernel_size= [5, 5],
                            stride = 2, scope = 'conv1')

        pool1 = slim.max_pool2d(inputs=conv1, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')

        conv2 = slim.conv2d(inputs = pool1, num_outputs = 64, kernel_size= [5, 5],
                            stride = 1, scope = 'conv2')

        pool2 = slim.max_pool2d(inputs=conv2, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool2')

        conv3 = slim.conv2d(inputs = pool2, num_outputs = 128, kernel_size= [3, 3],
                            stride = 1, scope = 'conv3')

        conv4 = slim.conv2d(inputs = conv3, num_outputs = 256, kernel_size= [3, 3],
                            stride = 1, scope = 'conv4')
        # full connect layer
        shp = conv4.get_shape()
        dim = shp[1].value * shp[2].value * shp[3].value
        conv4_flat = tf.reshape(conv4, [-1,dim])
        fc1 = slim.fully_connected(inputs=conv4_flat, num_outputs=256,activation_fn=tf.nn.relu, scope='fc1')
        fc1_drop = tf.nn.dropout(fc1, keep_prob)  # dropout layer
        fc1_drop_reshape = tf.reshape(fc1_drop, [-1, 15, 256])

        # with tf.variable_scope( reuse=True):
        lstm = rnn.LSTMCell(num_units=128, forget_bias=0.5)
        state = lstm.zero_state(batch_size, tf.float32)
        output_all =[]

        for time_step in range(15):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            output, state = lstm(fc1_drop_reshape[:, time_step, :], state)
            if (time_step + 1) % 3 == 0:
                output_all.append(output)

        output_all = tf.concat(output_all, 1)

        # tvars = tf.trainable_variables()

        print_activations(conv1)
        print_activations(pool1)
        print_activations(conv2)
        print_activations(pool2)
        print_activations(conv3)
        print_activations(conv4)
        print_activations(fc1_drop)
        print_activations(output_all)


    # with tf.variable_scope('fc2'):
    #     fc2 = slim.fully_connected(inputs=cell_output, num_outputs=1, activation_fn=None, scope='fc2')
    # return tf.reshape(fc2, [-1])
    return output_all

def main(_):
    #Graph
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 15* 4 * 101 * 101])    # input data
        y_ = tf.placeholder(tf.float32, [None,])    # label
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 4, 101, 101, 1])
        # x_image = tf.transpose(x_image, [0,3,4,1,2])
        x_image_h0 = x_image[:, 0, :, :, :]
        x_image_h1 = x_image[:, 1, :, :, :]
        x_image_h2 = x_image[:, 2, :, :, :]
        x_image_h3 = x_image[:, 3, :, :, :]

    print(x_image_h1.shape)
    fc1_h0 = net(x_image_h0, 0, keep_prob)
    fc1_h1 = net(x_image_h1, 1, keep_prob)
    fc1_h2 = net(x_image_h2, 2, keep_prob)
    fc1_h3 = net(x_image_h3, 3, keep_prob)

    fc1_all = tf.concat([fc1_h0, fc1_h1, fc1_h2, fc1_h3], axis=1)
    print(fc1_all.shape)
    fc2 = slim.fully_connected(inputs=fc1_all, num_outputs=20, activation_fn=None, scope='fc2')
    y_conv = slim.fully_connected(inputs=fc2, num_outputs=1, activation_fn=None, scope='fc3')
    y_conv = tf.reshape(y_conv, [-1])

    # loss function
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_conv, y_))))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


    # gpu configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.visible_device_list = '0'
    # gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


    with tf.Session(config=tf_config) as sess:

        saver = tf.train.Saver(max_to_keep = None)
        # Initialize parameters
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, dir_load)

        #Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", timestamp))
        print("Writing to {}\n".format(out_dir))

        #summary for loss
        tf.summary.scalar("loss", loss)
        tf.summary.histogram("fc1_all", fc1_all)
        # tf.summary.scalar("learning rate", learning_rate)
        # loss_summary = tf.summary.scalar("learning rate", learning_rate)

        # tf.summary.image("input", x_image_h1, max_outputs = 15)
        merged = tf.summary.merge_all()  # it is useful
        # grad_summaries = []
        # grad_summaries.append(loss_summary)
        # merged = tf.summary.merge(grad_summaries)

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

        #training
        count = 0
        for i in range(4):
            train_batches = data_process.batch_iter1(train_file, batch_size, shuffle=True)
            for batch in train_batches:
                count += 1
                # vari_x.extend(batch[2])
                # vari_y.extend(batch[3])
                #training with a batch
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})
                #training loss
                train_summary, train_loss_out = sess.run([merged, loss], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
                print("epochs %d, step %d, training loss %g" % (i, count * batch_size, train_loss_out))
                train_summary_writer.add_summary(train_summary, count)

                #varification
                # if (count * batch_size) % 3000 == 0:
                if count % 30 == 0:
                    veri_batches = data_process.batch_iter1(veri_file, veri_size)
                    testOutAll = []
                    y_all = []
                    for veri_batch in veri_batches:
                        vari_summary, veri_loss_out, y_veri= sess.run([merged, loss, y_conv], feed_dict={x: veri_batch[0], y_: veri_batch[1], keep_prob: 1})
                        y_all = np.concatenate([y_all, veri_batch[1]])
                        testOutAll = np.concatenate([testOutAll, y_veri])
                    # print(y_all.shape, testOutAll.shape)
                    err_2 = (testOutAll - y_all) ** 2
                    rmse = math.sqrt( np.mean(err_2) )
                    print("*******epochs %d, step %d, varification loss %g" % (i, count * batch_size, rmse))
                    dev_summary_writer.add_summary(vari_summary, count)

                #save model
                # if (count * batch_size) % 3000 == 0:
                if count % 30 == 0:
                    # if not os.path.exists(dir_model):
                    path = saver.save(sess, checkpoint_prefix, global_step = count * batch_size)
                    print("Saved model checkpoint to {}\n".format(path))
                    #     os.makedirs(dir_model)
                    # saver.save(sess, (dir_model + "/epochs%d_%d.ckpt") % (i, count * batch_size))


if __name__ == "__main__":
    tf.app.run()
