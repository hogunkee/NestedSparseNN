from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./sample/MNIST_data', one_hot=True)
import tensorflow as tf

### input ###
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

### function definition ###
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def make_filter_2(channel_in, channel_out):
    initial_1 = tf.truncated_normal([3, 3, channel_in, channel_out], stddev=0.1)
    initial_2 = tf.truncated_normal([3, 3, channel_out, channel_out], stddev=0.1)
    return tf.Variable(initial_1), tf.Variable(initial_2)

def make_filter_3(channel_in, channel_out):
    initial_1 = tf.truncated_normal([3, 3, channel_in, channel_out], stddev=0.1)
    initial_2 = tf.truncated_normal([3, 3, channel_out, channel_out], stddev=0.1)
    initial_3 = tf.truncated_normal([3, 3, channel_out, channel_out], stddev=0.1)
    return tf.Variable(initial_1), tf.Variable(initial_2), tf.Variable(initial_3)

def make_bias_2(channel):
    initial_1 = tf.constant(0.1, shape=[channel])
    initial_2 = tf.constant(0.1, shape=[channel])
    return tf.Variable(initial_1), tf.Variable(initial_2)

def make_bias_3(channel):
    initial_1 = tf.constant(0.1, shape=[channel])
    initial_2 = tf.constant(0.1, shape=[channel])
    initial_3 = tf.constant(0.1, shape=[channel])
    return tf.Variable(initial_1), tf.Variable(initial_2), tf.Variable(initial_3)

def conv(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_maxpool(x, filter_list, bias_list):
    for i in range(len(filter_list)):
        w = filter_list[i]
        b = bias_list[i]
        x = tf.nn.relu(conv(x, w) + b)
    x = maxpool(x)
    return x

### weight & bias setting ###
W1 = list(make_filter_2(1, 64)) # for MNIST
#W1 = list(make_filter_2(3, 64))
B1 = list(make_bias_2(64))
W2 = list(make_filter_2(64, 128))
B2 = list(make_bias_2(128))
W3 = list(make_filter_3(128, 256))
B3 = list(make_bias_3(256))
W4 = list(make_filter_3(256, 512))
B4 = list(make_bias_3(512))
W5 = list(make_filter_3(512, 512))
B5 = list(make_bias_3(512))
w_fc1 = weight_variable([512, 512])
b_fc1 = bias_variable([512])
w_fc2 = weight_variable([512, 512])
b_fc2 = bias_variable([512])
w_fc3 = weight_variable([512, 10])
b_fc3 = bias_variable([10])

x = tf.reshape(X, [-1,28,28,1]) # for MNIST
#x = tf.reshape(X, [-1,32,32,1])
h1 = conv_maxpool(x, W1, B1)
h2 = conv_maxpool(h1, W2, B2)
h3 = conv_maxpool(h2, W3, B3)
h4 = conv_maxpool(h3, W4, B4)
h5 = conv_maxpool(h4, W5, B5)
h5_flat = tf.reshape(h5, [-1, 512])
h_fc1 = tf.nn.relu(tf.matmul(h5_flat, w_fc1) + b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
y = tf.matmul(h_fc2, w_fc3) + b_fc3
#y = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)
#h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)
#y = tf.nn.softmax(h_fc3)

### train & evaluate ###
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
#loss = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
accur = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_loss, train_accur = sess.run([loss, accur], feed_dict={X:batch[0], Y:batch[1]})
        '''
        train_accuracy=accur.eval(feed_dict={X:batch[0],Y:batch[1]})
        '''
        print("step %d, training accuracy %g, loss %g"%(i, train_accur, train_loss))
    train_step.run(feed_dict={X:batch[0],Y:batch[1]})

print("test accuracy %g" %accur.eval(feed_dict={
    X:mnist.test.images,Y: mnist.test.labels}))

