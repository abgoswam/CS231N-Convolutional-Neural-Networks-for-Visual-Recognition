# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:54:24 2018

@author: agoswami
"""

import tensorflow as tf
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
v2 = tf.get_variable("bconv1", shape=[32])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "cifar_convnet_model/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())