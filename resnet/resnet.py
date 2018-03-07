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

### convolution and maxpooling function ###
def conv(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding='SAME')

def conv_bn_relu(x, dim1, dim2, tag, is_training):
    if dim1 != dim2:
        stride = 2
    else:
        stride = 1
    w = make_variable('Weight-' + tag, [3, 3, dim1, dim2])

    x_conv = conv(x, w, stride)
    x_bn = batch_norm(x_conv, dim2, is_training)
    x_relu = tf.nn.relu(x_bn)
    return x_relu

def res_block(x, dim1, dim2, scope, tag, is_training):
    with tf.variable_scope(scope):
        shortcut = x
        if dim1 != dim2:
            w = make_variable('shortcut-' + tag, [3,3,dim1,dim2])
            shorcut = conv(x, w, 2)

        x_1 = conv_bn_relu(x, dim1, dim2, tag + '-1', is_training)
        x_2 = conv_bn_relu(x, dim2, dim2, tag + '-2', is_training)

        return tf.add(x_2, x)

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_maxpool(x, filter_list, bias_list, scope, is_training):
    for i in range(len(filter_list)):
        w = filter_list[i]
        b = bias_list[i]
        x = tf.nn.relu(conv(x, w) + b)
        #print(x.shape[3])
        x = batch_norm(x, int(x.shape[3]), scope, is_training)
    x = maxpool(x)
    return x

def batch_norm(x, n_out, is_training = True):
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
        self.lr = config.learning_rate
        #self.learning_rate = config.learning_rate
        self.beta = config.beta
        self.image_size = config.image_size 

        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.print_step = config.print_step
        self.is_training = is_training
        n = self.n = 2


        self.X = X = tf.placeholder(tf.float32, shape = [None, 3*(self.image_size**2)])
        self.Y = Y = tf.placeholder(tf.float32, shape = [None, self.num_classes])

        noise = tf.constant([mean_RGB for i in range(self.batch_size)])

        x = tf.reshape(X, [-1, 32, 32, 3])
        #x = tf.reshape((X - noise), [-1, 32, 32, 3])
        '''
        x_flip = tf.map_fn(lambda k: tf.image.random_flip_left_right(k), x, dtype = tf.float32)
        paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
        x_pad = tf.pad(x_flip, paddings, 'CONSTANT')
        x_padcrop = tf.map_fn(lambda k: tf.random_crop(k, [32,32,3]), x_pad, dtype = tf.float32)
        '''
        
        # 2개씩 pad & crop

        dim1 = 16
        h1 = conv_bn_relu(x, 3, dim1, 'first', self.is_training)

        h2 = h1
        for i in range(n):
            h2 = res_block(h2, dim1, dim1, 'layer1', str(i), self.is_training)
        
        h3 = h2
        dim2 = 32
        for i in range(n):
            h3 = res_block(h3, dim1, dim2, 'layer2', str(i), self.is_training)
            dim1 = dim2

        h4 = h3
        dim2 = 64 
        for i in range(n):
            h4 = res_block(h4, dim1, dim2, 'layer3', str(i), self.is_training)
            dim1 = dim2

        h_reshape = tf.reshape(h4, [-1, 64, 64])
        h_max_pool = tf.reduce_max(h_reshape, 1)

        w_fc, b_fc = make_Wb_tuple(64, 10, 'fc')
        y = tf.matmul(h_max_pool, w_fc) + b_fc

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
        regularizer = tf.nn.l2_loss(w_fc)
        '''
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
        '''
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

