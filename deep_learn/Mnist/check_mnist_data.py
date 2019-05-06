# -*- coding: utf-8 -*-
import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# download Mnist data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# check the data shape
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)

print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# show one picture
a = mnist.train.images[1]
b = a.reshape(28, 28)
plt.imshow(b)

'''Save 20 pictures'''
save_dir = 'picture/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
    
for i in range(20):
    image_array = mnist.train.images[i]
    image_array = image_array.reshape(28, 28)
    filename = save_dir + 'mnist_train%d.jpg'%(i)
    #把array转换为图像
    scipy.misc.toimage(image_array).save(filename)

