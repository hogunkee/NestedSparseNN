import numpy as np
import tensorflow as tf

### RGB mean value ###
mean_R, mean_G, mean_B = 124.2, 123.4, 123.7
std = 69.68
rgb = 123.767
mean_RGB = [[rgb, rgb, rgb] for i in range(32*32)]
#mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]

### VGGNet model ###
class SparseResNet(object):
    def __init__(self, config, is_training = False):
        self.num_classes = config.num_classes
        self.num_classes2 = config.num_classes2

        self.dataset = config.dataset
        self.lr = config.learning_rate
        self.lr2 = config.learning_rate2
        self.beta = config.beta
        self.padding = config.padding
        self.norm = config.norm

        self.image_size = 32
        self.input_channel = 3
        n = self.n = config.num_layers

        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.print_step = config.print_step
        self.is_training = is_training


        self.X = X =  tf.placeholder(tf.float32, shape = [None, self.input_channel*(self.image_size**2)], name = 'X_placeholder')
        self.Y1 = Y1 = tf.placeholder(tf.float32, shape = [None, self.num_classes], name = 'Y1_placeholder')
        self.Y2 = Y2 = tf.placeholder(tf.float32, shape = [None, self.num_classes2], name = 'Y2_placeholder')

        self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')
        self.learning_rate2 = tf.placeholder(tf.float32, [], name = 'learning_rate2')

        x = tf.reshape(X, [-1, self.image_size, self.image_size, self.input_channel])
        ### pixel normalization ###
        '''
        print('Image standardization')
        x = tf.map_fn(lambda k: tf.image.per_image_standardization(k), x, dtype=tf.float32)
        '''

        ### flip, crop and padding ###
        if self.is_training==True:
            print('Training Model')
            print('image randomly flip')
            x = tf.map_fn(lambda k: tf.image.random_flip_left_right(k), x, dtype = tf.float32)

        def crop_pad(image):
            image = tf.image.resize_image_with_crop_or_pad(image, self.image_size+4, self.image_size+4)
            image = tf.random_crop(image, [self.image_size, self.image_size, 3])
            return image

        if self.is_training==True:
            print('image crop and padding')
            x = tf.map_fn(lambda k: crop_pad(k), x, dtype=tf.float32)
        '''
        if self.padding=='True' and self.is_training==True:
            print('image crop and padding')
            x = tf.map_fn(lambda k: tf.random_crop(
                tf.image.pad_to_bounding_box(k, 4, 4, 40, 40), [32, 32, 3]), 
                x, dtype = tf.float32)
        '''

        #first layer
        h1lv1, h2lv2 = self.first_layer(x)

        # c 16
        h1lv1, h2lv1, h2lv2 = self.res_block(h1lv1, h1lv1, h2lv2, 16, 'b1-layer-'+str(0), True)
        for i in range(1,n):
            h1lv1, h2lv1, h2lv2 = self.res_block(h1lv1, h2lv1, h2lv2, 16, 'b1-layer-'+str(i))

        # c 32
        for i in range(n):
            h1lv1, h2lv1, h2lv2 = self.res_block(h1lv1, h2lv1, h2lv2, 32, 'b2-layer-'+str(i))

        # c 64
        for i in range(n):
            h1lv1, h2lv1, h2lv2 = self.res_block(h1lv1, h2lv1, h2lv2, 64, 'b3-layer-'+str(i))

        h1lv1 = self.batch_norm(h1lv1, h1lv1.shape[3], 'fc-lv1')
        h2lv1 = self.batch_norm(h2lv1, h2lv1.shape[3], 'fc-lv2-l1')
        h2lv2 = self.batch_norm(h2lv2, h2lv2.shape[3], 'fc-lv2-l2')
        h1lv1 = self.relu(h1lv1)
        h2lv1 = self.relu(h2lv1)
        h2lv2 = self.relu(h2lv2)
        h1lv1 = tf.reduce_mean(h1lv1, [1,2])
        h2lv1 = tf.reduce_mean(h2lv1, [1,2])
        h2lv2 = tf.reduce_mean(h2lv2, [1,2])

        lv1 = h1lv1
        lv2 = tf.concat((h2lv1, h2lv2), 1)

        y1 = self.fc(lv1, self.num_classes, 'fc-lv1')
        y2 = self.fc(lv2, self.num_classes2, 'fc-lv2')
        self.out1 = tf.nn.softmax(y1)

        self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y1,logits=y1))
        self.loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y2,logits=y2))
        self.loss_t = self.loss1 + self.loss2

        correct_predict1 = tf.equal(tf.argmax(y1,1), tf.argmax(Y1,1))
        correct_predict2 = tf.equal(tf.argmax(y2,1), tf.argmax(Y2,1))
        self.accur1 = tf.reduce_mean(tf.cast(correct_predict1, tf.float32))
        self.accur2 = tf.reduce_mean(tf.cast(correct_predict2, tf.float32))

        t_vars = tf.trainable_variables()
        self.l1_vars = [v for v in t_vars if 'lv1' in v.name] + [v for v in t_vars if 'input' in
                v.name]
        self.l2_vars = [v for v in t_vars if not ('lv1' in v.name or 'input' in v.name)]
        #self.l2_vars = self.l1_vars + [v for v in t_vars if 'lv2' in v.name]

        # print vars
        from pprint import pprint
        #if self.is_training:
            #pprint(t_vars)
            #pprint(self.l1_vars)
            #pprint(self.l2_vars)

        self.regularizer1 = self.l2loss(self.l1_vars)
        self.regularizer2 = self.l2loss(self.l2_vars)

        self.loss1 += self.beta * self.regularizer1
        self.loss2 += self.beta * self.regularizer2
        self.loss_t += self.beta * self.regularizer2

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            '''
            optimizer = tf.train.AdamOptimizer(self.learning_rate, name='1')
            optimizer2 = tf.train.AdamOptimizer(self.learning_rate2, name='2')
            optimizer_t = tf.train.AdamOptimizer(self.learning_rate, name='t')
            '''
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, name='1', use_nesterov=True)
            optimizer2 = tf.train.MomentumOptimizer(self.learning_rate2, 0.9, name='2', use_nesterov=True)
            optimizer_t = tf.train.MomentumOptimizer(self.learning_rate, 0.9, name='t', use_nesterov=True)
            self.train_step1 = optimizer.minimize(self.loss1, var_list=self.l1_vars)
            self.train_step2 = optimizer2.minimize(self.loss2, var_list=self.l2_vars)
            self.train_step_t = optimizer_t.minimize(self.loss_t, var_list=t_vars)


    ### functions ###
    def first_layer(self, x):
        with tf.variable_scope('input'):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            #c_init = tf.contrib.layers.xavier_initializer()
            n = np.sqrt(6/(3*3*3*16))
            c_init = tf.random_uniform_initializer(-n, n)
            b_init = tf.constant_initializer(0.0)

            out1 = tf.contrib.layers.conv2d(x, 12, [3,3], activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init)
            out2 = tf.contrib.layers.conv2d(x, 4, [3,3], activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init)
        return out1, out2

    def res_block(self, x11, x21, x22, out, scope, activate_before_residual=False):
        num_in = int(x11.shape[3]) + int(x22.shape[3])
        if out//num_in == 2:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            if activate_before_residual:
                x11_bn = self.batch_norm(x11, x11.shape[3], 'lv1-res1')
                x21_bn = self.batch_norm(x21, x21.shape[3], 'lv2-l1-res1')
                x22_bn = self.batch_norm(x22, x22.shape[3], 'lv2-l2-res1')
                x11_relu = self.relu(x11_bn)
                x21_relu = self.relu(x21_bn)
                x22_relu = self.relu(x22_bn)
                shortcut11 = x11_relu
                shortcut21 = x21_relu
                shortcut22 = x22_relu
                x11 = self.lv1conv(x11_relu, out*3/4, stride, 'lv1-res1')
                x21, x22 = self.lv2conv(x21_relu, x22_relu, out*3/4, out/4, stride, 'lv2-res1')
                x21 = tf.add(x11, x21)
            else:
                shortcut11 = x11
                shortcut21 = x21
                shortcut22 = x22
                x11_bn = self.batch_norm(x11, x11.shape[3], 'lv1-res1')
                x21_bn = self.batch_norm(x21, x21.shape[3], 'lv2-l1-res1')
                x22_bn = self.batch_norm(x22, x22.shape[3], 'lv2-l2-res1')
                x11_relu = self.relu(x11_bn)
                x21_relu = self.relu(x21_bn)
                x22_relu = self.relu(x22_bn)
                x11 = self.lv1conv(x11_relu, out*3/4, stride, 'lv1-res1')
                x21, x22 = self.lv2conv(x21_relu, x22_relu, out*3/4, out/4, stride, 'lv2-res1')
                x21 = tf.add(x11, x21)

            x11_bn = self.batch_norm(x11, x11.shape[3], 'lv1-res2')
            x21_bn = self.batch_norm(x21, x21.shape[3], 'lv2-l1-res2')
            x22_bn = self.batch_norm(x22, x22.shape[3], 'lv2-l2-res2')
            x11_relu = self.relu(x11_bn)
            x21_relu = self.relu(x21_bn)
            x22_relu = self.relu(x22_bn)
            x11 = self.lv1conv(x11_relu, out*3/4, 1, 'lv1-res2')
            x21, x22 = self.lv2conv(x21_relu, x22_relu, out*3/4, out/4, 1, 'lv2-res2')
            x21 = tf.add(x11, x21)

            if int(num_in) != out:
                shortcut11 = self.shortcut(shortcut11, out*3/4, 'lv1-sc')
                shortcut21 = self.shortcut(shortcut21, out*3/4, 'lv2-l1-sc')
                shortcut22 = self.shortcut(shortcut22, out/4, 'lv2-l2-sc')

            return tf.add(x11, shortcut11), tf.add(x21, shortcut21), tf.add(x22, shortcut22)

    def lv1conv(self, x, dim, stride, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            n = np.sqrt(6/(4 * 4 * int(x.shape[3]) * dim))
            c_init = tf.random_uniform_initializer(-n, n)
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            #c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)

            out = tf.contrib.layers.conv2d(x, dim, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init)
        return out

    def lv2conv(self, x1, x2, dim1, dim2, stride, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            n = np.sqrt(6 / (3 * 3 * int(x1.shape[3] + x2.shape[3]) * (dim1 + dim2)))
            c_init = tf.random_uniform_initializer(-n, n)
            #c_init = tf.contrib.layers.xavier_initializer()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            b_init = tf.constant_initializer(0.0)

            concat_x = tf.concat((x1, x2), 3)

            out1 = tf.contrib.layers.conv2d(x2, dim1, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init, scope='l1')
            out2 = tf.contrib.layers.conv2d(concat_x, dim2, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init, scope='l2')
        return out1, out2

    def fc(self, x, dim, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #f_init = tf.truncated_normal_initializer(stddev=5e-2)
            #f_init = tf.contrib.layers.xavier_initializer()
            f_init = tf.uniform_unit_scaling_initializer(factor=1.0)
            b_init = tf.constant_initializer(0.0)
            return tf.contrib.layers.fully_connected(x, dim, activation_fn=None,
                    weights_initializer=f_init, biases_initializer=b_init)

    def shortcut(self, x, num_out, scope):
        input_c = x.shape[3]
        pool = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.pad(pool, [[0,0], [0,0], [0,0], [input_c//2, input_c//2]])

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

    def maxpool(self, x, scope):
        return tf.contrib.layers.max_pool2d(x, [2,2], stride=2, padding='SAME', scope=scope)

    def l2loss(self, var_list):
        regul = tf.nn.l2_loss(var_list[0])
        for v in var_list[1:]:
            regul += tf.nn.l2_loss(v)
        return regul
