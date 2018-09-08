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

def get_batch(x, y, batch_size):
#    print("x shape : {0}".format(x.shape))
#    print("y shape : {0}".format(y.shape))
    n_samples = x.shape[0]
    indices = np.random.choice(n_samples, batch_size)
    return x[indices], y[indices], indices

# clear old variables
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="Input")
y = tf.placeholder(tf.int64, [None])

# Placeholders for batchnorm and dropout
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

# setup variables
Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
bconv1 = tf.get_variable("bconv1", shape=[32])
W1 = tf.get_variable("W1", shape=[5408, 10])
b1 = tf.get_variable("b1", shape=[10])

# define our graph (e.g. two_layer_convnet)
a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
h1 = tf.nn.relu(a1, name="OutputConvRelu_1")
h1_flat = tf.reshape(h1,[-1,5408])
y_out = tf.matmul(h1_flat,W1,name="Output") + b1

# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, depth=10), logits=y_out))

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_history = []
train_accuracy_history = []
val_accuracy_history = []

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = get_batch(X_train, y_train, 50)
        if i % 400 == 0:
            train_accuracy = accuracy.eval(feed_dict={X: batch[0], y: batch[1], keep_prob: 1.0, is_training:False})
            val_accuracy = accuracy.eval(feed_dict={X: X_val, y: y_val, keep_prob: 1.0, is_training:False})
            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(val_accuracy)
            print('step %d, training accuracy %g, , validation accuracy %g' % (i, train_accuracy, val_accuracy))
                  
        _, loss_i = sess.run([train_step, cross_entropy], feed_dict={X: batch[0], y: batch[1], keep_prob: 0.75, is_training:True})
        loss_history.append(loss_i)

    print('test accuracy %g' % accuracy.eval(feed_dict={X: X_val, y: y_val, keep_prob: 1.0, is_training:False}))
    
    # Save the variables to disk.
#    save_path = saver.save(sess, "cifar_save9/model.ckpt")
#    print("Model saved in path: %s" % save_path)
    
    export_dir = "cifar_save10"
    tf.saved_model.simple_save(sess,
            export_dir,
            inputs={"Input": X},
            outputs={"Output": y_out})
    



# Run this cell to visualize training loss and train / val accuracy

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(train_accuracy_history, '-o', label='train')
plt.plot(val_accuracy_history, '-o', label='val')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

for op in tf.get_default_graph().get_operations():
    print(op.name)

