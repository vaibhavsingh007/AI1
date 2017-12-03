# Fetch MNIST data (ref: http://yann.lecun.com/exdb/mnist/)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Create placeholder for input vector to the model
# 'None' implies unbounded dimension
# 784 is input image pixels dim.
x = tf.placeholder(tf.float32, [None, 784])

# Define weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# The model
y = tf.matmul(x, W) + b

# Calculating cross-entropy to minimize loss
y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# Read more about cross-entropy story here: 
# http://colah.github.io/posts/2015-09-Visual-Information/
# https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize the variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Let's train the model
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))