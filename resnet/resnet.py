# https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220770760226&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F

import tensorflow as tf

### function to make variable ###
def make_variable(name, size):
    init = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, size, dtype=tf.float32, initializer=init)

def make_bias(name, size):
    init = tf.zeros_initializer()
    return tf.get_variable(name, size, dtype=tf.float32, initializer=init)

def make_Wb_tuple(dim1, dim2, tag):
    w = make_variable('Weight-' + tag, [dim1, dim2])
    b = make_bias('bias-' + tag, [dim2])
    return w, b

def make_Wb_list(dim1, dim2, tag, num_layer):
    W_list = []
    B_list = []
    for num in range(num_layer):
        w = make_variable('Weight' + tag + '-' + str(num), [3, 3, dim1, dim2])
        b = make_variable('bias' + tag + '-' + str(num), [dim2])
        W_list.append(w)
        B_list.append(b)
        dim1 = dim2
    return W_list, B_list

### convolution and maxpooling function ###
def conv(x,w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_maxpool(x, filter_list, bias_list, scope, is_training):
    for i in range(len(filter_list)):
        '''
        if not i==0:
            x = tf.nn.dropout(x, 0.4)
        '''
        w = filter_list[i]
        b = bias_list[i]
        x = tf.nn.relu(conv(x, w) + b)
        #print(x.shape[3])
        x = batch_norm(x, int(x.shape[3]), scope, is_training)
    x = maxpool(x)
    return x

def batch_norm(x, n_out, scope, is_training = True):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay = 0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return batch_mean, batch_var

        if is_training:
            mean, var = mean_var_with_update()
        else:
            mean, var = batch_mean, batch_var
            #mean, var = ema.average(batch_mean), ema.average(batch_var)
            #print(mean,var)

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed

def batch_norm2(x, n_out, scope, is_training = True):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay = 0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return batch_mean, batch_var

        if is_training:
            mean, var = mean_var_with_update()
        else:
            mean, var = batch_mean, batch_var
            #mean, var = ema.average(batch_mean), ema.average(batch_var)
            #print(mean,var)

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed

### RGB mean value ###
mean_R, mean_G, mean_B = 125.3, 122.9, 113.9
std = 64.15
mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]

### ResNet model ###
class ResNet(object):
    def __init__(self, config, is_training = False):
        self.num_classes = config.num_classes
        self.keep_prob = 1 - config.dropout
        self.lr = config.learning_rate
        #self.learning_rate = config.learning_rate
        self.beta = config.beta
        self.image_size = config.image_size 

        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.print_step = config.print_step
        self.is_training = is_training


        self.X = X = tf.placeholder(tf.float32, shape = [None, 3*(self.image_size**2)])
        self.Y = Y = tf.placeholder(tf.float32, shape = [None, self.num_classes])

        '''
        if is_training == False:
            self.keep_prob = 1.0
        '''

        with tf.variable_scope('ResNet'):
            if not is_training:
                tf.get_variable_scope().reuse_variables()
                #print(tf.get_variable_scope().name)
            self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')
            W1, B1 = make_Wb_list(3, 64, '1', 2)
            W2, B2 = make_Wb_list(64, 128, '2', 2)
            W3, B3 = make_Wb_list(128, 256, '3', 3)
            W4, B4 = make_Wb_list(256, 512, '4', 3)
            W5, B5 = make_Wb_list(512, 512, '5', 3)
            w_fc1, b_fc1 = make_Wb_tuple(512, 512, 'fc1')
            w_fc2, b_fc2 = make_Wb_tuple(512, 512, 'fc2')
            w_fc3, b_fc3 = make_Wb_tuple(512, 10, 'fc3')
            noise = tf.constant([mean_RGB for i in range(self.batch_size)])

        x = tf.reshape(X, [-1, 32, 32, 3])
        #x = tf.reshape((X - noise), [-1, 32, 32, 3])
        x_flip = tf.map_fn(lambda k: tf.image.random_flip_left_right(k), x, dtype = tf.float32)
        paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
        x_pad = tf.pad(x_flip, paddings, 'CONSTANT')
        x_padcrop = tf.map_fn(lambda k: tf.random_crop(k, [32,32,3]), x_pad, dtype = tf.float32)
        #x_padcrop = tf.map_fn(lambda k: tf.random_crop(tf.image.pad_to_bounding_box(k, 2, 2, 36, 36), [32, 32, 3]), x_flip, dtype = tf.float32)
        #self.sample = x_padcrop[0]
        
        # 2개씩 pad & crop

        ## convolution & maxpooling layer ##
        h1 = conv_maxpool(x_padcrop, W1, B1, '1', self.is_training)
        #h1 = conv_maxpool(x, W1, B1, '1', self.is_training)
        h2 = conv_maxpool(h1, W2, B2, '2', self.is_training)
        h3 = conv_maxpool(h2, W3, B3, '3', self.is_training)
        h4 = conv_maxpool(h3, W4, B4, '4', self.is_training)
        h5 = conv_maxpool(h4, W5, B5, '5', self.is_training)

        ## fully connected layer ##
        h5_flat = tf.reshape(h5, [-1, 512])
        h_fc1 = tf.nn.relu(tf.matmul(h5_flat, w_fc1) + b_fc1)
        #h_norm1 = h_fc1
        h_norm1 = batch_norm2(h_fc1, int(h_fc1.shape[1]), 'fc1', self.is_training)
        h_dropout1 = tf.nn.dropout(h_norm1, self.keep_prob)
        h_fc2 = tf.nn.relu(tf.matmul(h_dropout1, w_fc2) + b_fc2)
        #h_norm2 = h_fc2
        h_norm2 = batch_norm2(h_fc2, int(h_fc2.shape[1]), 'fc2', self.is_training)
        h_dropout2 = tf.nn.dropout(h_norm2, self.keep_prob)
        y = tf.matmul(h_dropout2, w_fc3) + b_fc3

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

        self.regularizer = regularizer
        self.loss = loss = loss + self.beta * self.regularizer

        '''
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
        '''
        #train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        #train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
            train_step = optimizer.minimize(loss)
            self.train_step = train_step

        correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
        self.accur = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

