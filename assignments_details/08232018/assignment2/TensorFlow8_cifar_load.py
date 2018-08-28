# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:49:30 2018

@author: agoswami
"""

import argparse
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="cifar_save8/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    x = graph.get_tensor_by_name('prefix/Input:0')
    y = graph.get_tensor_by_name('prefix/OutputConvRelu_1:0')

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: X_train # < 45
        })
        print(y_out) # [[ False ]] Yay!