# https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220770760226&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F

import tensorflow as tf

### RGB mean value ###
mean_R, mean_G, mean_B = 125.3, 122.9, 113.9
std = 64.15
mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]

### ResNet model ###
class ResNet(object):
    def __init__(self, config, is_training = False):
        self.num_classes = config.num_classes
        self.dataset = config.dataset
        self.lr = config.learning_rate
        self.beta = config.beta
        self.padding = config.padding
        self.norm = config.norm

        if self.dataset=='mnist':
            self.image_size = 28
            self.input_channel = 1
        elif self.dataset=='cifar10' or self.dataset=='cifar100':
            self.image_size = 32
            self.input_channel = 3
        else:
            self.image_size = 224 
            self.input_channel = 3

        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.print_step = config.print_step
        self.is_training = is_training
        n = self.n = config.num_layers


        self.X = X = tf.placeholder(tf.float32, shape = [None, 3*(self.image_size**2)])
        self.Y = Y = tf.placeholder(tf.float32, shape = [None, self.num_classes])

        self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')

        noise = tf.constant([mean_RGB for i in range(self.batch_size)])

        ### pixel normalization ###
        if self.dataset=='cifar10' and self.norm=='True':
            print('pixel normalization')
            noise = tf.constant([mean_RGB for i in range(self.batch_size)])
            x = tf.reshape((X-noise)/std, [-1, self.image_size, self.image_size, self.input_channel])
        else:
            x = tf.reshape(X, [-1, self.image_size, self.image_size, self.input_channel])

        ### flip, crop and padding ###
        if self.is_training==True:
            print('Training Model')
            print('image randomly flip')
            x = tf.map_fn(lambda k: tf.image.random_flip_left_right(k), x, dtype = tf.float32)

        if self.padding=='True' and self.is_training==True:
            print('image crop and padding')
            x = tf.map_fn(lambda k: tf.random_crop(
                tf.image.pad_to_bounding_box(k, 4, 4, 40, 40), [32, 32, 3]), 
                x, dtype = tf.float32)
        
        h = self.conv_bn_relu(x, 16, 'first')

        for i in range(n):
            h = self.res_block(h, 16, 'layer1-'+str(i))

        for i in range(n):
            h = self.res_block(h, 32, 'layer2-'+str(i))

        for i in range(n):
            h = self.res_block(h, 64, 'layer3-'+str(i))

        h = tf.reshape(h, [-1, 8*8, 64])
        h = tf.reduce_mean(h, 1)
        #h = tf.reduce_max(h, 1)
        y = self.fc(h, 10, 'fc-layer')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += sum(reg_losses)

        self.loss = loss
        self.regularizer = sum(reg_losses)

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
            train_step = optimizer.minimize(loss)
            self.train_step = train_step

        correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
        self.accur = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    ### convolution, maxpooling and fully-connected function ###
    def conv(self, x, num_out, stride, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
            return tf.contrib.layers.conv2d(x, num_out, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init, weights_regularizer=regularizer,
                    biases_initializer=b_init)

    def fc(self, x, num_out, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #f_init = tf.truncated_normal_initializer(stddev=5e-2)
            f_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
            return tf.contrib.layers.fully_connected(x, num_out, activation_fn=None,
                    weights_initializer=f_init, weights_regularizer=regularizer,
                    biases_initializer=b_init)

    def batch_norm(self, x, num_out, scope):
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[num_out]),
                    name='beta', trainable=self.is_training)
            gamma = tf.Variable(tf.constant(1.0, shape=[num_out]),
                    name='gamma', trainable=self.is_training)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay = 0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return batch_mean, batch_var

            if self.is_training:
                mean, var = mean_var_with_update()
            else:
                mean, var = batch_mean, batch_var

            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
        return normed

    def conv_bn_relu(self, x, num_out, scope):
        if num_out//int(x.shape[3])==2:
            stride = 2
        else:
            stride = 1

        x_conv = self.conv(x, num_out, stride, scope)
        x_bn = self.batch_norm(x_conv, num_out, scope)
        x_relu = tf.nn.relu(x_bn)
        return x_relu

    def shortcut(self, x, num_out, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
            return tf.contrib.layers.conv2d(x, num_out, [1,1], 2, activation_fn=None, 
                    weights_initializer=c_init, weights_regularizer=regularizer,
                    biases_initializer=b_init)

    '''
    def shortcut2(self, x, num_out, scope):
        input_c = x.shape[3]
        pool = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.pad(pool, [[0,0], [0,0], [0,0], [input_c//2, input_c//2]])
    '''


    def res_block(self, x, num_out, scope):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            shortcut = x
            if int(x.shape[3]) != num_out:
                shortcut = self.shortcut(x, num_out, scope+'-sc')

            x_1 = self.conv_bn_relu(x, num_out, scope + '-1')
            x_2 = self.conv_bn_relu(x_1, num_out, scope + '-2')

            return tf.add(x_2, shortcut)
