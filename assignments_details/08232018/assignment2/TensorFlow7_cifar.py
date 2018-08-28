# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:05:56 2018

@author: agoswami
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

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

# Lets get a 1000 samples from the data
batch_size = 1000
n_samples = X_train.shape[0]
indices = np.random.choice(n_samples, batch_size)
train_data = X_train[indices]
train_labels = y_train[indices]

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 32, 32, 3], name="input_tensor")

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[7, 7],
      strides=(2, 2),
      padding="valid",
      activation=tf.nn.relu,
      name="conv1_tensor")

  # # Pooling Layer #1
  # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  #
  # # Convolutional Layer #2 and Pooling Layer #2
  # conv2 = tf.layers.conv2d(
  #     inputs=pool1,
  #     filters=64,
  #     kernel_size=[5, 5],
  #     padding="same",
  #     activation=tf.nn.relu)
  # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  #
  # Dense Layer
  pool2_flat = tf.reshape(conv1, [-1, 5408])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10, name="logits_tensor")

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
def main(unused_argv):
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images  # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  #eval_data = mnist.test.images  # Returns np.array
  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  cifar_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="cifar_convnet_model_2")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)


  print("=============================")

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=100,
      shuffle=True)
  
  cifar_classifier.train(
      input_fn=train_input_fn,
      steps=200,
      hooks=[logging_hook])

  print("++++++++++++++++++++++++++++")
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      num_epochs=1,
      shuffle=False)
  
  eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  print("##########################")
  
        
if __name__ == "__main__":
  tf.app.run()