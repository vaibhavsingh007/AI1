import DatasetUtils as du
import tensorflow as tf
import numpy as np

xy_train = du.load_data('target', 'train.csv')
xy_test = du.load_data('target', 'test.csv')

x_train = xy_train[0]
y_train = xy_train[1]
x_test = xy_test[0]
y_test = xy_test[1]

# Weight Initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and Pooling functions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Conv1
W_conv1 = weight_variable([1, 1, 1, 32])
b_conv1 = bias_variable([32])

x = tf.placeholder(tf.float32, [None, 93])

# Reshaping x dimensions to match Weight's
x = tf.reshape(x, [-1, 1, 93, 1])

# Convolute with Weight and then Maxpool
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Conv3
W_conv2 = weight_variable([1, 1, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([1 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Read out (or Fully Connected layer 2)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_ = tf.placeholder(tf.float32, [None, 10])

# Train and Evaluate
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_train = x_train.as_matrix()
    x_train = np.expand_dims(np.expand_dims(x_train, 0), -1)
    y_train = y_train.eval()
    x_test = x_test.as_matrix()
    x_test = np.expand_dims(np.expand_dims(x_test, 0), -1)
    y_test = y_test.eval()

    for i in range(x_train.shape[0]-1):
        train_accuracy = accuracy.eval(feed_dict={
            x: [x_train[i]], y_: [y_train[i]], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: [x_train[i]], y_: [y_train[i]], keep_prob: 0.5})

    x = tf.reshape(x, [-1, len(x_test[0]), 93, 1])
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: x_test, y_: y_test, keep_prob: 1.0}))