import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

export_dir = "hellotensor_2"

with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
  
  graph = tf.get_default_graph()
  print(graph.get_operations())
    
  a_test = sess.run('O:0', feed_dict={'I:0': [[1, 2, 3],[4, 5, 6]]})
  print("test acc : {0}".format(a_test))