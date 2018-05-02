# https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220770760226&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F

import numpy as np
import tensorflow as tf

### RGB mean value ###
#mean_R, mean_G, mean_B = 125.3, 122.9, 113.9
std = 64.15
#mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]
rgb = 120.707
mean_RGB = [[rgb, rgb, rgb] for i in range(32*32)]

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

        x = tf.reshape(X, [-1, self.image_size, self.image_size, self.input_channel])
        ### pixel normalization ###
        '''
        if self.dataset=='cifar10' and self.norm=='True':
            print('pixel normalization')
            noise = tf.constant([mean_RGB for i in range(self.batch_size)])
            noise_tensor = tf.reshape(noise, [-1, self.image_size, self.image_size, self.input_channel])
            x = (x - noise_tensor)/std
        '''

        #print('Image stadardization')
        #x = tf.map_fn(lambda k: tf.image.per_image_standardization(k), x, dtype=tf.float32)

        ### flip, crop and padding ###
        if self.is_training==True:
            print('Training Model')
            print('image randomly flip')
            x = tf.map_fn(lambda k: tf.image.random_flip_left_right(k), x, dtype = tf.float32)

        def crop_pad(image):
            image = tf.image.resize_image_with_crop_or_pad(image, self.image_size+4, self.image_size+4)
            image = tf.random_crop(image, [self.image_size, self.image_size, 3])
            return image

        if self.padding=='True' and self.is_training==True:
            print('image crop and padding')
            x = tf.map_fn(lambda k: crop_pad(k), x, dtype=tf.float32)
            '''
            x = tf.map_fn(lambda k: tf.random_crop(
                tf.image.pad_to_bounding_box(k, 4, 4, 40, 40), [32, 32, 3]), 
                x, dtype = tf.float32)
            '''
        
        h = self.conv(x, 16, 1, 'first')
        #h = self.conv_bn_relu(x, 16, 'first')

        for i in range(n):
            h = self.res_block(h, 16, 'layer1-'+str(i), True)

        for i in range(n):
            h = self.res_block(h, 32, 'layer2-'+str(i))

        for i in range(n):
            h = self.res_block(h, 64, 'layer3-'+str(i))

        num_in = int(h.shape[3])
        h = self.batch_norm(h, num_in, 'fc')
        h = self.relu(h)
        h = tf.reduce_mean(h, [1,2])
        #h = tf.reduce_max(h, 1)
        y = self.fc(h, 10, 'fc-layer')

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))

        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        t_vars = tf.trainable_variables()
        self.regularizer = self.l2loss(t_vars)

        self.loss += self.beta * self.regularizer

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            #optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.0)
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov=True)
            train_step = optimizer.minimize(self.loss)
            self.train_step = train_step

        correct_predict = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
        self.accur = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    ### convolution, maxpooling and fully-connected function ###
    def conv(self, x, num_out, stride, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            #c_init = tf.contrib.layers.xavier_initializer()
            #c_init = tf.random_normal_initializer(stddev=np.sqrt(2.0/(9*num_out)))
            n = np.sqrt(6.0 / (3 * 3 * int(x.shape[3]) * num_out))
            c_init = tf.random_uniform_initializer(-n, n)
            b_init = tf.constant_initializer(0.0)
            return tf.contrib.layers.conv2d(x, num_out, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=None)
                    #weights_initializer=c_init, biases_initializer=b_init)
            '''
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
            return tf.contrib.layers.conv2d(x, num_out, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init, weights_regularizer=regularizer,
                    biases_initializer=b_init)
            '''

    def fc(self, x, num_out, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #f_init = tf.truncated_normal_initializer(stddev=5e-2)
            #f_init = tf.contrib.layers.xavier_initializer()
            f_init = tf.uniform_unit_scaling_initializer(factor=1.0)
            b_init = tf.constant_initializer(0.0)
            return tf.contrib.layers.fully_connected(x, num_out, activation_fn=None,
                    weights_initializer=f_init, biases_initializer=b_init)
            '''
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
            return tf.contrib.layers.fully_connected(x, num_out, activation_fn=None,
                    weights_initializer=f_init, weights_regularizer=regularizer,
                    biases_initializer=b_init)
            '''

    def batch_norm(self, input_layer, dimension, scope):
        BN_DECAY = 0.999 #0.9
        BN_EPSILON = 1e-5 #1e-3
        with tf.variable_scope(scope): 
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            beta = tf.get_variable('beta', dimension, tf.float32,
                         initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', dimension, tf.float32,
                         initializer=tf.constant_initializer(1.0, tf.float32))
            mu = tf.get_variable('mu', dimension, tf.float32,
                         initializer=tf.constant_initializer(0.0, tf.float32))
            sigma = tf.get_variable('sigma', dimension, tf.float32,
                         initializer=tf.constant_initializer(1.0, tf.float32))
     
            if self.is_training is True:
                mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
                train_mean = tf.assign(mu, mu*BN_DECAY + mean*(1-BN_DECAY))
                train_var = tf.assign(sigma, sigma*BN_DECAY + variance*(1 - BN_DECAY))
         
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON    )
            else:
                bn_layer = tf.nn.batch_normalization(input_layer, mu, sigma, beta, gamma, BN_EPSILON)
         
            return bn_layer

    def relu(self, x):
        leakiness = 0.1
        return tf.where(tf.less(x, 0.0), leakiness*x, x, name='leaky_relu')

    def conv_bn_relu(self, x, num_out, scope):
        if num_out//int(x.shape[3])==2:
            stride = 2
        else:
            stride = 1

        x_conv = self.conv(x, num_out, stride, scope)
        x_bn = self.batch_norm(x_conv, num_out, scope)
        x_relu = self.relu(x_bn)
        return x_relu

    def bn_relu_conv(self, x, num_out, scope):
        num_in = int(x.shape[3])
        if num_out//num_in == 2:
            stride = 2
        else:
            stride = 1
        x_bn = self.batch_norm(x, num_in, scope)
        x_relu = self.relu(x_bn)
        x_conv = self.conv(x_relu, num_out, stride, scope)
        return x_conv

    # not used
    def shortcut2(self, x, num_out, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta)
            short_x = tf.contrib.layers.conv2d(x, num_out, [1,1], 2, activation_fn=None, 
                    weights_initializer=c_init, weights_regularizer=regularizer,
                    biases_initializer=b_init)
            return short_x

    def shortcut(self, x, num_out, scope):
        input_c = x.shape[3]
        pool = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.pad(pool, [[0,0], [0,0], [0,0], [input_c//2, input_c//2]])

    def res_block(self, x, num_out, scope, activate_before_residual=False):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            num_in = int(x.shape[3])
            if activate_before_residual:
                x_bn = self.batch_norm(x, num_in, scope+'-1')
                x_relu = self.relu(x_bn)
                shortcut = x_relu
                x_1 = self.conv(x_relu, num_out, 1, scope+'-1')
            else:
                shortcut = x
                x_1 = self.bn_relu_conv(x, num_out, scope+'-1')

            x_2 = self.bn_relu_conv(x_1, num_out, scope+'-2')

            if int(num_in) != num_out:
                shortcut = self.shortcut(x, num_out, scope+'-sc')

            return tf.add(x_2, shortcut)

    def l2loss(self, var_list):
        regul = tf.nn.l2_loss(var_list[0])
        for v in var_list[1:]:
            regul += tf.nn.l2_loss(v)
        return regul
