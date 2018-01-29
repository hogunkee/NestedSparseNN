# needed: validation, regularization, learning rate decrease
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./sample/MNIST_data', one_hot=True)
'''
import random
import pickle
import os
import os.path
import numpy as np
import tensorflow as tf

### config ###
num_epoch = 20
batch_size = 50
learning_rate = 1e-5
num_labels = 10
validation = 0.1
beta = 5e-4
print('config')
print('num epoch: %d' %(num_epoch))
print('batch size: %d' %(batch_size))
print('learning_rate: %g' %(learning_rate))
print('validation split: %g' %(validation))
print('regularization rate: %g' %(beta))


### data loading ###
def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'labels'], dict[b'data']

def one_hot(label):
    zero = [0 for _ in range(num_labels)]
    for i in range(len(label)):
        tmp = zero.copy()
        tmp[label[i]] = 1
        label[i] = tmp

path = '../data/cifar-10-batches-py/'

train_labels = []
train_data = []
test_labels = []
test_data = []
for fname in os.listdir(path):
    fpath = os.path.join(path, fname)
    _label, _data = unpickle(fpath)
    print('load', fname)
    if fname == 'test_batch': 
        test_labels = _label
        test_data = _data
    else:
        if train_labels==[]:
            train_labels = _label
            train_data = _data
        else:
            train_labels = train_labels + _label
            train_data = np.concatenate((train_data, _data))
#print(train_data[0][:30])

### RGB mean value ###
mean_R, mean_G, mean_B = 125.3, 122.9, 113.9
mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]
'''
sum_R, sum_G, sum_B, num_data = 0, 0, 0, 0
for data in train_data:
    sum_R += sum(data[:32*32])/(32*32)
    sum_G += sum(data[32*32:2*32*32])/(32*32)
    sum_B += sum(data[2*32*32:])/(32*32)
    num_data +=1
mean_R = sum_R / num_data
mean_G = sum_G  /num_data
mean_B = sum_B / num_data
print(mean_R, mean_G, mean_B)
'''
tmp = list(zip(train_data, train_labels))
random.shuffle(tmp)
train_data, train_labels = zip(*tmp)

data_train = list(train_data)[:int(-validation * len(train_data))]
labels_train = list(train_labels)[:int(-validation * len(train_labels))]
data_val = list(train_data)[-int(validation * len(train_data)):]
labels_val = list(train_labels)[-int(validation * len(train_labels)):]
data_test = test_data
labels_test = test_labels

one_hot(labels_train)
one_hot(labels_val)
one_hot(labels_test)
print('train data length: %d' %(len(labels_train)))
print('validation data length: %d' %(len(labels_val)))
print('test data length: %d' %(len(labels_test)))

### input ###
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32, shape=[None, 3*32*32])
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
W1 = list(make_filter_2(3, 64))
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
#noise = tf.zeros([batch_size, 3*32*32], tf.float32)
noise = tf.constant([mean_RGB for i in range(batch_size)])
#noise = tf.reshape(noise, [-1, 32, 32, 3])

x = tf.reshape(X - noise, [-1,32,32,3])
#x = tf.reshape(X, [-1,32,32,3])
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
regularizer = tf.nn.l2_loss(w_fc1)
regularizer += tf.nn.l2_loss(w_fc2)
regularizer += tf.nn.l2_loss(w_fc3)
for w in W1:
    regularizer += tf.nn.l2_loss(w)
for w in W2:
    regularizer += tf.nn.l2_loss(w)
for w in W3:
    regularizer += tf.nn.l2_loss(w)
for w in W4:
    regularizer += tf.nn.l2_loss(w)
for w in W5:
    regularizer += tf.nn.l2_loss(w)
loss = loss + beta * regularizer
#loss = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
accur = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
sess.run(tf.global_variables_initializer())

### training ###
pre_val = 0
for epoch in range(num_epoch):
    print("epoch %d" % (epoch+1))
    for i in range(len(data_train)//batch_size):
        input_data = data_train[batch_size * i : batch_size * (i+1)]
        input_label = labels_train[batch_size * i : batch_size * (i+1)]
        if (i+1)%100==0:
            train_loss, train_accur = sess.run([loss, accur], feed_dict={X:input_data, Y:input_label})
            print("step %d, training accuracy %g, loss %g"%(i+1, train_accur, train_loss))
        train_step.run(feed_dict={X:input_data,Y:input_label})

    ### validation data accuracy ###
    sum_accur, num_data = 0, 0
    for i in range(len(data_val)//batch_size):
        input_data = data_val[batch_size * i : batch_size * (i+1)]
        input_label = labels_val[batch_size * i : batch_size * (i+1)]
        val_loss, val_accur = sess.run([loss, accur], feed_dict={X:input_data, Y:input_label})
        if (i+1)%100==0:
            print("step %d, test accuracy %g, loss %g"%(i+1, val_accur, val_loss))
        sum_accur += val_accur
        num_data += 1
    curr_val = sum_accur / num_data
    print("validation accuracy %g"%(curr_val))
    if (curr_val < pre_val):
        learning_rate /= 10
        print('change learning rate %g:' %(learning_rate))
    pre_val = curr_val

### test data accuracy ###
sum_accur, num_data = 0, 0
for i in range(len(data_test)//batch_size):
    input_data = data_test[batch_size * i : batch_size * (i+1)]
    input_label = labels_test[batch_size * i : batch_size * (i+1)]
    test_loss, test_accur = sess.run([loss, accur], feed_dict={X:input_data, Y:input_label})
    if (i+1)%100==0:
        print("step %d, test accuracy %g, loss %g"%(i+1, test_accur, test_loss))
    sum_accur += test_accur
    num_data += 1

print("final test accuracy %g"%(sum_accur / num_data))

'''
print("test accuracy %g" %accur.eval(feed_dict={
    X:data_test,Y:labels_test}))
'''
