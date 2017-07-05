#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""writing recognition in tensorflow"""
"""
the net has four inputs and takes a concat operation on time sequence(the 15 moment)
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

veri_size = 100
batch_size = 100
dir_load = "/home/zxf/PycharmProjects/CIKM/CNN_new/run/1498603335/checkpoints/model-33000"

def print_activations(t):
    print(t.op.name, t.get_shape().as_list())

def net(image, i_th, keep_prob):
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

def main(_):
    #Graph
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 15* 4 * 101 * 101])    # input data
        y_ = tf.placeholder(tf.float32, [None,])    # label
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 15, 4, 101, 101, 1])
        # x_image = tf.transpose(x_image, [0,3,4,1,2])
        x_image_h0 = x_image[:, 14, 0, :, :, :]
        x_image_h1 = x_image[:, 14, 1, :, :, :]
        x_image_h2 = x_image[:, 14, 2, :, :, :]
        x_image_h3 = x_image[:, 14, 3, :, :, :]

    print(x_image_h1.shape)
    fc2_h0 = net(x_image_h0, 0, keep_prob)
    fc2_h1 = net(x_image_h1, 1, keep_prob)
    fc2_h2 = net(x_image_h2, 2, keep_prob)
    fc2_h3 = net(x_image_h3, 3, keep_prob)

    fc1_all = tf.concat([fc2_h0, fc2_h1, fc2_h2, fc2_h3], axis=1)
    print(fc1_all.shape)
    y_conv = slim.fully_connected(inputs=fc1_all, num_outputs=1, activation_fn=tf.nn.relu, scope='fc3')
    y_conv = tf.reshape(y_conv, [-1])

    # loss function
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_conv, y_))))

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(1e-4, global_step, 89, 0.5, staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_step = optimizer.minimize(loss, global_step=global_step)

    train_step = tf.train.AdamOptimizer(5e-5).minimize(loss)

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

        #summary for loss
        tf.summary.scalar("loss", loss)
        # tf.summary.scalar("lossWithl2", lossWithl2)
        tf.summary.histogram("fc1_all", fc1_all)
        # tf.summary.scalar("learning rate", learning_rate)

        tf.summary.image("input_h0", x_image_h0, max_outputs = 3)
        tf.summary.image("input_h1", x_image_h1, max_outputs = 3)
        tf.summary.image("input_h2", x_image_h2, max_outputs = 3)
        tf.summary.image("input_h3", x_image_h3, max_outputs = 3)
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
                #training with a batch
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
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
