# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:41:41 2019

@author: dh
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# get data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# parameters
epoch_num = 20
batch_num = 200
batch_size = mnist.train.images.shape[0] // batch_num

# define the input placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# variable
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# operation
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

# coss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred)))

# train
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# accuracy
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))  # True or False
# transfor to accuraacy
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

# create session and run
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(epoch_num):
    # get batch data
    for batch in range(batch_num):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x:batch_x, y:batch_y})
    validation_accuracy = sess.run(accuracy, feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
    print("accuracy of epoch %d is %f"%(epoch, validation_accuracy))
    
# test use test data
accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
print("accuracy is %.3f"%accuracy)