import tensorflow as tf
import DatasetUtils as du

xy_train = du.load_data('target', 'train.csv')
xy_test = du.load_data('target', 'test.csv')

x_train = xy_train[0]
y_train = xy_train[1]
x_test = xy_test[0]
y_test = xy_test[1]

# Create placeholder for input vector to the model
# 'None' implies unbounded dimension
# 93 is input feature dim.
x = tf.placeholder(tf.float32, [None, 93])

# Define weights and biases
W = tf.Variable(tf.zeros([93, 10]))
b = tf.Variable(tf.zeros([10]))

# The model
y = tf.matmul(x, W) + b

# Calculating cross-entropy to minimize loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# Read more about cross-entropy story here:
# http://colah.github.io/posts/2015-09-Visual-Information/
# https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize the variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

x_train = x_train.as_matrix()
y_train = y_train.eval()
x_test = x_test.as_matrix()
y_test = y_test.eval()

# Let's train the model
for k in range(100):    # Adjust this achieve the desired accuracy and/or efficiency
    sess.run(train_step, feed_dict={x: x_train, y_: y_train})
    print("Training step: %d" % k)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy on training data:")
print(sess.run(accuracy, feed_dict={x: x_train, y_: y_train}))

print("Classifying test data:")
t = sess.run(tf.argmax(y,1), feed_dict={x: x_test})
print(t)

# Uncomment following to enumerate all classifications
#for i in range(len(t)-1):
#    print(x_test[i])
#    print(t[i])


