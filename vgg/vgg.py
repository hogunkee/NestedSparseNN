import tensorflow as tf

### RGB mean value ###
mean_R, mean_G, mean_B = 125.3, 122.9, 113.9
std = 64.15
mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]

### VGGNet model ###
class VGG(object):
    def __init__(self, config, is_training = False):
        self.num_classes = config.num_classes
        self.dataset = config.dataset
        self.keep_prob = 1 - config.dropout
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

        if not self.is_training:
            self.keep_prob = 1.0


        self.X = X = tf.placeholder(tf.float32, shape = [None, self.input_channel*(self.image_size**2)])
        self.Y = Y = tf.placeholder(tf.float32, shape = [None, self.num_classes])

        self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')

        ### pixel normalization ###
        if self.dataset=='cifar10' and self.norm==True:
            print('pixel normalization')
            noise = tf.constant([mean_RGB for i in range(self.batch_size)])
            x = tf.reshape((X-noise)/std, [-1, self.image_size, self.image_size, self.input_channel])
        else:
            x = tf.reshape(X, [-1, self.image_size, self.image_size, self.input_channel])

        ### flip, crop and padding ###
        if self.is_training==True:
            x = tf.map_fn(lambda k: tf.image.random_flip_left_right(k), x, dtype = tf.float32)

        if self.padding==True and self.is_training==True:
            print('image crop and padding')
            x = tf.map_fn(lambda k: tf.random_crop(
                tf.image.pad_to_bounding_box(k, 4, 4, 40, 40), [32, 32, 3]), 
                x, dtype = tf.float32)

        #channel: 64
        h1 = self.conv(x, 64, 'layer-1')
        h1 = self.batch_norm(h1, 64, 'layer-1')
        h1 = tf.nn.relu(h1)
        h2 = self.conv(h1, 64, 'layer-2')
        h2 = self.batch_norm(h2, 64, 'layer-2')
        h2 = tf.nn.relu(h2)
        h2 = self.maxpool(h2, 'layer-2')
        #channel: 128
        h3 = self.conv(h2, 128, 'layer-3')
        h3 = self.batch_norm(h3, 128, 'layer-3')
        h3 = tf.nn.relu(h3)
        h4 = self.conv(h3, 128, 'layer-4')
        h4 = self.batch_norm(h4, 128, 'layer-4')
        h4 = tf.nn.relu(h4)
        h4 = self.maxpool(h4, 'layer-4')
        #channel: 256
        h5 = self.conv(h4, 256, 'layer-5')
        h5 = self.batch_norm(h5, 256, 'layer-5')
        h5 = tf.nn.relu(h5)
        h6 = self.conv(h5, 256, 'layer-6')
        h6 = self.batch_norm(h6, 256, 'layer-6')
        h6 = tf.nn.relu(h6)
        h7 = self.conv(h6, 256, 'layer-7')
        h7 = self.batch_norm(h7, 256, 'layer-7')
        h7 = tf.nn.relu(h7)
        h7 = self.maxpool(h7, 'layer-7')
        #channel: 512
        h8 = self.conv(h7, 512, 'layer-8')
        h8 = self.batch_norm(h8, 512, 'layer-8')
        h8 = tf.nn.relu(h8)
        h9 = self.conv(h8, 512, 'layer-9')
        h9 = self.batch_norm(h9, 512, 'layer-9')
        h9 = tf.nn.relu(h9)
        h10 = self.conv(h9, 512, 'layer-10')
        h10 = self.batch_norm(h10, 512, 'layer-10')
        h10 = tf.nn.relu(h10)
        h10 = self.maxpool(h10, 'layer-10')
        #channel: 512
        h11 = self.conv(h10, 512, 'layer-11')
        h11 = self.batch_norm(h11, 512, 'layer-11')
        h11 = tf.nn.relu(h11)
        h12 = self.conv(h11, 512, 'layer-12')
        h12 = self.batch_norm(h12, 512, 'layer-12')
        h12 = tf.nn.relu(h12)
        h13 = self.conv(h12, 512, 'layer-13')
        h13 = self.batch_norm(h13, 512, 'layer-13')
        h13 = tf.nn.relu(h13)
        h13 = self.maxpool(h13, 'layer-13')
        #fc layer 512, 512, 10
        h14 = tf.reshape(h13, [-1, 512])
        h14 = self.fc(h14, 512, 'fc-1')
        h14 = tf.nn.dropout(h14, self.keep_prob)
        h14 = self.batch_norm2(h14, 512, 'layer-14')
        h14 = tf.nn.relu(h14)
        h15 = self.fc(h14, 512, 'fc-2')
        h15 = tf.nn.dropout(h15, self.keep_prob)
        h15 = self.batch_norm2(h15, 512, 'layer-15')
        h15 = tf.nn.relu(h15)
        y = self.fc(h15, 10, 'fc-3')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += sum(reg_losses)
        self.loss = loss
        self.regularizer = sum(reg_losses)

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
            train_step = optimizer.minimize(loss, var_list = tf.trainable_variables())

        self.train_step = train_step
        correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
        self.accur = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    ### convolution, maxpooling and fully-connected function ###
    def conv(self, x, num_out, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
            return tf.contrib.layers.conv2d(x, num_out, [3,3], activation_fn=None, 
                    weights_initializer=c_init, weights_regularizer=regularizer,
                    biases_initializer=b_init)

    def maxpool(self, x, scope):
        return tf.contrib.layers.max_pool2d(x, [2,2], stride=2, padding='SAME', scope=scope)
    
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
                    name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[num_out]),
                    name='gamma', trainable=True)
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

    def batch_norm2(self, x, num_out, scope):
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[num_out]),
                    name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[num_out]),
                    name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
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
