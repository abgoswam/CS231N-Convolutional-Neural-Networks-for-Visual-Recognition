# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 18:31:15 2018

@author: agoswami
"""

import tensorflow as tf
import mnistdata

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, 784])
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0 
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={X: batch_X, Y_: batch_Y}

    # train
    sess.run(train_step, feed_dict=train_data)
    a_train,c_test = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    
    # success on test data ?
    test_data={X: mnist.test.images, Y_: mnist.test.labels}
    a_test,c_test = sess.run([accuracy, cross_entropy], feed=test_data)
    
    print("i:{0} a_train : {1} a_test : {2}".format(i, a_train, a_test))

