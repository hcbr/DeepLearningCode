# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:54:41 2019

@author: dh
"""

'''
this model use two convolution layers, two max pool layers
and two fully connected layers to recongize the Mnist data.
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# get data 
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# parameters
epoch_num = 20
batch_num = 200
batch_size = mnist.train.images.shape[0] // batch_num

# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

# reshape
x_image = tf.reshape(x, [-1, 28, 28, 1])


## define four function for fenerating parameters

# W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# conv
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


# pooling
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


## first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

## second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)
#print(h_pool2)    ## check the shape (7,7)

## fully connected layer(flatten)
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# flatten
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully connected layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

## compute loss
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_conv)
        )

## define train_step
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

## compute accuracy
y_pred = tf.nn.softmax(y_conv)
correct_predict = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

## create Session for trainning
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(epoch_num):
    for batch in range(batch_num):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict = {x:batch_x, y:batch_y, keep_prob:1})
    # validate
    validation_accuracy = sess.run(
            accuracy, feed_dict = {x:mnist.validation.images, y:mnist.validation.labels, keep_prob:1}
            )
    print("the accuracy of epoch%d is %.3f"%(epoch, validation_accuracy))
    
# test result
accuracy = sess.run(
        accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1}
        )
print("the test accuracy is %.3f"%accuracy)

















