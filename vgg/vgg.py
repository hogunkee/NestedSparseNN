import tensorflow as tf

### RGB mean value ###
mean_R, mean_G, mean_B = 125.3, 122.9, 113.9
mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]

### VGGNet model ###
class VGG(object):
    def __init__(self, config, is_training = False):
        self.num_classes = config.num_classes
        self.keep_prob = 1 - config.dropout
        self.lr = config.learning_rate
        #self.learning_rate = config.learning_rate
        self.beta = config.beta
        self.is_training = is_training
        if self.dataset == 'mnist':
            self.image_size = 28
            self.input_channel = 1
        else:
            self.image_size = 32
            self.input_channel = 3

        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.print_step = config.print_step


        self.X = X = tf.placeholder(tf.float32, shape = [None, self.input_channel*(self.image_size**2)])
        self.Y = Y = tf.placeholder(tf.float32, shape = [None, self.num_classes])

        '''
        if is_training == False:
            self.keep_prob = 1.0
        '''

        '''
        with tf.variable_scope('VGG'):
            if not is_training:
                tf.get_variable_scope().reuse_variables()
            self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')
            W1, B1 = make_Wb_list(self.input_channel, 64, '1', 2)
            W2, B2 = make_Wb_list(64, 128, '2', 2)
            W3, B3 = make_Wb_list(128, 256, '3', 3)
            W4, B4 = make_Wb_list(256, 512, '4', 3)
            W5, B5 = make_Wb_list(512, 512, '5', 3)
            w_fc1, b_fc1 = make_Wb_tuple(512, 512, 'fc1')
            w_fc2, b_fc2 = make_Wb_tuple(512, 512, 'fc2')
            w_fc3, b_fc3 = make_Wb_tuple(512, 10, 'fc3')
            noise = tf.constant([mean_RGB for i in range(self.batch_size)])
        '''

        x = tf.reshape(X, [-1, self.image_size, self.image_size, self.input_channel])
        #x = tf.reshape(X - noise, [-1, self.image_size, self.image_size, self.input_channel])
        x_flip = tf.map_fn(lambda k: tf.image.random_flip_left_right(k), x, dtype = tf.float32)
        x_padcrop = tf.map_fn(lambda k: tf.random_crop(
            tf.image.pad_to_bounding_box(k, 4, 4, 40, 40), [32, 32, 3]), 
            x_flip, dtype = tf.float32)

        #channel: 64
        if self.padding:
            h1 = conv(x_padcrop, 64, 'layer-1')
        else: 
            h1 = conv(x, 64, 'layer-1')
        h1 = tf.nn.relu(h1)
        h2 = conv(h1, 64, 'layer-2')
        h2 = tf.nn.relu(h2)
        #channel: 128
        h3 = maxpool(h2, 'layer-3')
        h3 = conv(h3, 128, 'layer-3')
        h3 = tf.nn.relu(h3)
        h4 = conv(h3, 128, 'layer-4')
        h4 = tf.nn.relu(h4)
        #channel: 256
        h5 = maxpool(h4, 'layer-5')
        h5 = conv(h5, 256, 'layer-5')
        h5 = tf.nn.relu(h5)
        h6 = conv(h6, 256, 'layer-6')
        h6 = tf.nn.relu(h6)
        h7 = conv(h7, 256, 'layer-7')
        h7 = tf.nn.relu(h7)
        #channel: 512
        h8 = maxpool(h7, 'layer-8')
        h8 = conv(h8, 512, 'layer-8')
        h8 = tf.nn.relu(h8)
        h9 = conv(h8, 512, 'layer-9')
        h9 = tf.nn.relu(h9)
        h10 = conv(h9, 512, 'layer-10')
        h10 = tf.nn.relu(h10)
        #channel: 512
        h11 = maxpool(h10, 'layer-11')
        h11 = conv(h11, 512, 'layer-11')
        h11 = tf.nn.relu(h11)
        h12 = conv(h11, 512, 'layer-12')
        h12 = tf.nn.relu(h12)
        h13 = conv(h12, 512, 'layer-13')
        h13 = tf.nn.relu(h13)
        #fc layer 512, 512, 10
        h14 = tf.reshape(h13, [-1, 512])
        h14 = fc(h14, 512, 'fc-1')
        h14 = tf.nn.relu(h14)
        h15 = fc(h14, 512, 'fc-2')
        h15 = tf.nn.relu(h15)
        y = fc(h15, 10, 'fc-3')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += self.beta * sum(reg_losses)

        with tf.variable_scope('VGG'):
            if not is_training:
                tf.get_variable_scope().reuse_variables()
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.train_step = train_step

        correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
        self.accur = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    ### convolution, maxpooling and fully-connected function ###
    def conv(self, x, num_out, scope):
        c_init = tf.truncated_normal_initializer(stddev=5e-2)
        b_init = tf.constant_initializer(0.0)
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
        return tf.contrib.layers.conv2d(x, num_out, [3,3] activation_fn=None,
                weights_initializer=c_init, weights_regularizer=regularizer,
                biases_initializer=b_init, scope=scope)

    def maxpool(self, x, scope):
        return tf.contrib.layers.max_pool2d(x, [2,2], padding='SAME', scope=scope)
    
    def fc(self, x, num_out, scope):
        f_init = tf.truncated_normal_initializer(stddev=5e-2)
        b_init = tf.constant_initializer(0.0)
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
        return tf.contrib.layers.fully_connected(x, num_out, activation_fn=None,
                weights_initializer=f_init, weights_regularizer=regularizer,
                biases_initializer=b_init, scope=scope)
