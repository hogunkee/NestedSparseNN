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
        self.num_classes3 = config.num_classes3

        self.dataset = config.dataset
        self.lr = config.learning_rate
        self.lr2 = config.learning_rate2
        self.lr3 = config.learning_rate3
        self.beta = config.beta

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
        self.Y3 = Y3 = tf.placeholder(tf.float32, shape = [None, self.num_classes3], name = 'Y3_placeholder')

        self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')
        self.learning_rate2 = tf.placeholder(tf.float32, [], name = 'learning_rate2')
        self.learning_rate3 = tf.placeholder(tf.float32, [], name = 'learning_rate3')

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

        if self.is_training==True:
            print('image crop and padding')
            x = tf.map_fn(lambda k: tf.random_crop(
                tf.image.pad_to_bounding_box(k, 4, 4, 40, 40), [32, 32, 3]), 
                x, dtype = tf.float32)

        #first layer
        lv1, lv2, lv3 = self.first_layer(x)

        first_layers = [lv1, lv1, lv2, lv1, lv2, lv3]

        # c 16 
        ## h_layers = [h1lv1, h2lv1, h2lv2, h3lv1, h3lv2, h3lv3] ##
        h_layers = self.res_block(first_layers, 16, 'b1-layer-'+str(0), True)
        for i in range(1,n):
            h_layers = self.res_block(h_layers, 16, 'b1-layer-'+str(i))

        # c 32
        for i in range(n):
            h_layers = self.res_block(h_layers, 32, 'b2-layer-'+str(i))

        # c 64
        for i in range(n):
            h_layers = self.res_block(h_layers, 64, 'b3-layer-'+str(i))

        h1lv1 = h_layers[0]
        h2lv1 = h_layers[1]
        h2lv2 = h_layers[2]
        h3lv1 = h_layers[3]
        h3lv2 = h_layers[4]
        h3lv3 = h_layers[5]
        h1lv1 = self.layers_batch_norm(h_layers[:1], 'fc-lv1-l')[0]
        h2lv1, h2lv2 = self.layers_batch_norm(h_layers[1:3], 'fc-lv2-l')
        h3lv1, h3lv2, h3lv3 = self.layers_batch_norm(h_layers[3:], 'fc-lv3-l')

        h_layers = [h1lv1, h2lv1, h2lv2, h3lv1, h3lv2, h3lv3]
        h_layers = self.relu_global_pool(h_layers)

        lv1 = h_layers[0]
        lv2 = tf.concat(h_layers[1:3], 1)
        lv3 = tf.concat(h_layers[3:], 1)

        y1 = self.fc(lv1, self.num_classes, 'fc-lv1')
        y2 = self.fc(lv2, self.num_classes2, 'fc-lv2')
        y3 = self.fc(lv3, self.num_classes3, 'fc-lv3')
        
        self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y1,logits=y1))
        self.loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y2,logits=y2))
        self.loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y3,logits=y3))

        correct_predict1 = tf.equal(tf.argmax(y1,1), tf.argmax(Y1,1))
        correct_predict2 = tf.equal(tf.argmax(y2,1), tf.argmax(Y2,1))
        correct_predict3 = tf.equal(tf.argmax(y3,1), tf.argmax(Y3,1))
        self.accur1 = tf.reduce_mean(tf.cast(correct_predict1, tf.float32))
        self.accur2 = tf.reduce_mean(tf.cast(correct_predict2, tf.float32))
        self.accur3 = tf.reduce_mean(tf.cast(correct_predict3, tf.float32))

        t_vars = tf.trainable_variables()
        self.l1_vars = [v for v in t_vars if 'lv1' in v.name]
        self.l2_vars = [v for v in t_vars if 'lv2' in v.name]
        self.l3_vars = [v for v in t_vars if 'lv3' in v.name]

        assert len(t_vars) == len(self.l1_vars + self.l2_vars + self.l3_vars)

        # print vars
        from pprint import pprint
        #if self.is_training:
            #pprint(t_vars)
            #pprint(self.l1_vars)
            #pprint(self.l2_vars)

        self.regularizer1 = self.l2loss(self.l1_vars)
        self.regularizer2 = self.l2loss(self.l2_vars)
        self.regularizer3 = self.l2loss(self.l3_vars)

        self.loss1 += self.beta * self.regularizer1
        self.loss2 += self.beta * self.regularizer2
        self.loss3 += self.beta * self.regularizer3

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9, name='1', use_nesterov=True)
            optimizer2 = tf.train.MomentumOptimizer(self.learning_rate2, 0.9, name='2', use_nesterov=True)
            optimizer3 = tf.train.MomentumOptimizer(self.learning_rate3, 0.9, name='3', use_nesterov=True)
            self.train_step1 = optimizer.minimize(self.loss1, var_list = self.l1_vars)
            self.train_step2 = optimizer2.minimize(self.loss2, var_list = self.l2_vars)
            self.train_step3 = optimizer3.minimize(self.loss3, var_list = self.l3_vars)

    ### functions ###
    def first_layer(self, x):
        with tf.variable_scope('input'):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            #c_init = tf.contrib.layers.xavier_initializer()
            n = np.sqrt(6/(3*3*(3+16)))
            c_init = tf.random_uniform_initializer(-n, n)
            #b_init = tf.constant_initializer(0.0)

            out1 = tf.contrib.layers.conv2d(x, 12, [3,3], activation_fn=None, 
                    weights_initializer=c_init, scope='lv1')
            out2 = tf.contrib.layers.conv2d(x, 2, [3,3], activation_fn=None, 
                    weights_initializer=c_init, scope='lv2')
            out3 = tf.contrib.layers.conv2d(x, 2, [3,3], activation_fn=None, 
                    weights_initializer=c_init, scope='lv3')
        return out1, out2, out3

    def res_block(self, x_list, out, scope, activate_before_residual=False):
        num_in = int(x_list[3].shape[3]) + int(x_list[4].shape[3]) + int(x_list[5].shape[3])
        if out//num_in == 2:
            stride = 2
        else:
            stride = 1

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            if activate_before_residual:
                x11 = self.layers_batch_norm(x_list[:1], 'lv1-res1-')[0]
                x21, x22 = self.layers_batch_norm(x_list[1:3], 'lv2-res1-')
                x31, x32, x33 = self.layers_batch_norm(x_list[3:], 'lv3-res1-')

                x_list = [x11, x21, x22, x31, x32, x33]
                x_list = self.layers_relu(x_list)
                shortcut = x_list

                x11 = self.lv1conv(x_list[0], out*3/4, stride, 'lv1-res1-')
                x21, x22 = self.lv2conv(x_list[1], x_list[2], out*3/4, out/8, stride, 'lv2-res1-')
                x31, x32, x33 = self.lv3conv(x_list[3], x_list[4], x_list[5],
                        out*3/4, out/8, out/8, stride, 'lv3-res1-')

                x21 = tf.add(x11, x21)
                x31 = tf.add(x21, x31)
                x32 = tf.add(x22, x32)
            else:
                shortcut = x_list

                x11 = self.layers_batch_norm(x_list[:1], 'lv1-res1-')[0]
                x21, x22 = self.layers_batch_norm(x_list[1:3], 'lv2-res1-')
                x31, x32, x33 = self.layers_batch_norm(x_list[3:], 'lv3-res1-')

                x_list = [x11, x21, x22, x31, x32, x33]
                x_list = self.layers_relu(x_list)

                x11 = self.lv1conv(x_list[0], out*3/4, stride, 'lv1-res1-')
                x21, x22 = self.lv2conv(x_list[1], x_list[2], out*3/4, out/8, stride, 'lv2-res1-')
                x31, x32, x33 = self.lv3conv(x_list[3], x_list[4], x_list[5],
                        out*3/4, out/8, out/8, stride, 'lv3-res1-')

                x21 = tf.add(x11, x21)
                x31 = tf.add(x21, x31)
                x32 = tf.add(x22, x32)

            x_list = [x11, x21, x22, x31, x32, x33]
            x11 = self.layers_batch_norm(x_list[:1], 'lv1-res2-')[0]
            x21, x22 = self.layers_batch_norm(x_list[1:3], 'lv2-res2-')
            x31, x32, x33 = self.layers_batch_norm(x_list[3:], 'lv3-res2-')

            x_list = [x11, x21, x22, x31, x32, x33]
            x_list = self.layers_relu(x_list)

            x11 = self.lv1conv(x_list[0], out*3/4, 1, 'lv1-res2-')
            x21, x22 = self.lv2conv(x_list[1], x_list[2], out*3/4, out/8, 1, 'lv2-res2-')
            x31, x32, x33 = self.lv3conv(x_list[3], x_list[4], x_list[5],
                    out*3/4, out/8, out/8, 1, 'lv3-res2-')

            x21 = tf.add(x11, x21)
            x31 = tf.add(x21, x31)
            x32 = tf.add(x22, x32)

            if int(num_in) != out:
                shortcut = self.layers_shortcut(shortcut)

            x_list[0] = x11 + shortcut[0]
            x_list[1] = x21 + shortcut[1]
            x_list[2] = x22 + shortcut[2]
            x_list[3] = x31 + shortcut[3]
            x_list[4] = x32 + shortcut[4]
            x_list[5] = x33 + shortcut[5]

            return x_list

    def lv1conv(self, x, dim, stride, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            n = np.sqrt(6/(4 * 4 * (int(x.shape[3]) + dim)))
            c_init = tf.random_uniform_initializer(-n, n)
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            #c_init = tf.contrib.layers.xavier_initializer()
            #b_init = tf.constant_initializer(0.0)

            out = tf.contrib.layers.conv2d(x, dim, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init)
        return out

    def lv2conv(self, x1, x2, dim1, dim2, stride, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            n = np.sqrt(6/(4 * 4 * (int(x1.shape[3]) + dim1)))
            #n = np.sqrt(6 / (3 * 3 * int(x1.shape[3] + x2.shape[3]) * (dim1 + dim2)))
            c_init = tf.random_uniform_initializer(-n, n)
            #c_init = tf.contrib.layers.xavier_initializer()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            #b_init = tf.constant_initializer(0.0)

            concat_x = tf.concat((x1, x2), 3)

            out1 = tf.contrib.layers.conv2d(x2, dim1, [3,3], stride, 
                    activation_fn=None, weights_initializer=c_init, scope='l1')
            out2 = tf.contrib.layers.conv2d(concat_x, dim2, [3,3], stride, 
                    activation_fn=None, weights_initializer=c_init, scope='l2')
        return out1, out2

    def lv3conv(self, x1, x2, x3, dim1, dim2, dim3, stride, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            n = np.sqrt(6/(4 * 4 * (int(x1.shape[3]) + dim1)))
            #n = np.sqrt(6 / (3 * 3 * int(x1.shape[3] + x2.shape[3]) * (dim1 + dim2)))
            c_init = tf.random_uniform_initializer(-n, n)
            #c_init = tf.contrib.layers.xavier_initializer()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            #b_init = tf.constant_initializer(0.0)

            concat_x2 = tf.concat((x1, x2), 3)
            concat_x3 = tf.concat((x1, x2, x3), 3)

            out1 = tf.contrib.layers.conv2d(x2, dim1, [3,3], stride, activation_fn=None, 
                    weights_initializer=c_init, scope='l1')
            out2 = tf.contrib.layers.conv2d(concat_x2, dim2, [3,3], stride, 
                    activation_fn=None, weights_initializer=c_init, scope='l2')
            out3 = tf.contrib.layers.conv2d(concat_x3, dim3, [3,3], stride, 
                    activation_fn=None, weights_initializer=c_init, scope='l3')
        return out1, out2, out3

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

    def layers_shortcut(self, l_list):
        out = []
        for l in l_list:
            out.append(self.shortcut(l))
        return out

    def shortcut(self, x):
        input_c = x.shape[3]
        pool = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return tf.pad(pool, [[0,0], [0,0], [0,0], [input_c//2, input_c//2]])

    def layers_batch_norm(self, l_list, scope):
        out = []
        for i in range(len(l_list)):
            out.append(self.batch_norm(l_list[i], scope+str(i)))
        return out

    def batch_norm(self, input_layer, scope):
        BN_DECAY = 0.999 #0.9
        BN_EPSILON = 1e-5 #1e-3
        dimension = int(input_layer.shape[3])
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

    def layers_relu(self, l_list):
        out = []
        for l in l_list:
            out.append(self.relu(l))
        return out

    def relu(self, x):
        leakiness = 0.1
        return tf.where(tf.less(x, 0.0), leakiness*x, x, name='leaky_relu')

    def l2loss(self, var_list):
        regul = tf.nn.l2_loss(var_list[0])
        for v in var_list[1:]:
            regul += tf.nn.l2_loss(v)
        return regul

    def relu_global_pool(self, l_list):
        out = []
        for l in l_list:
            out.append(tf.reduce_mean(self.relu(l), [1,2]))
        return out
