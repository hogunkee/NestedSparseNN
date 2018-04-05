import tensorflow as tf

### RGB mean value ###
mean_R, mean_G, mean_B = 125.3, 122.9, 113.9
std = 64.15
mean_RGB = [mean_R for i in range(32*32)] + [mean_G for i in range(32*32)] + [mean_B for i in range(32*32)]

### VGGNet model ###
class SparseVGG(object):
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
        self.fc = config.fc

        if not self.is_training:
            self.keep_prob = 1.0


        self.X = X =  tf.placeholder(tf.float32, shape = [None, self.input_channel*(self.image_size**2)], name = 'X_placeholder')
        self.Y = Y = tf.placeholder(tf.float32, shape = [None, self.num_classes], name = 'Y_placeholder')

        self.learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')

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

        #input layer
        #channel: 48 & 16
        h1lv1, h2lv2 = self.first_layer(x)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #1st layer
        #channel: 48 & 16
        h1lv1 = lv1conv(h1lv1, 48, 'layer1-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 48, 16, 'layer1-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #2nd layer
        #channel: 48 & 16
        h1lv1 = lv1conv(h1lv1, 48, 'layer2-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 48, 16, 'layer2-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #max pool
        h1lv1 = self.maxpool(h1lv1, 'max-1')
        h2lv1 = self.maxpool(h2lv1, 'max-1')
        h2lv2 = self.maxpool(h2lv2, 'max-1')

        #3rd layer
        #channel: 96 & 32 
        h1lv1 = lv1conv(h1lv1, 96, 'layer3-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 96, 32, 'layer3-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #4th layer
        #channel: 96 & 32 
        h1lv1 = lv1conv(h1lv1, 96, 'layer4-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 96, 32, 'layer4-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #max pool
        h1lv1 = self.maxpool(h1lv1, 'max-2')
        h2lv1 = self.maxpool(h2lv1, 'max-2')
        h2lv2 = self.maxpool(h2lv2, 'max-2')

        #5th layer
        #channel: 192 & 64 
        h1lv1 = lv1conv(h1lv1, 192, 'layer5-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 192, 64, 'layer5-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #6th layer
        #channel: 192 & 64 
        h1lv1 = lv1conv(h1lv1, 192, 'layer6-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 192, 64, 'layer6-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #7th layer
        #channel: 192 & 64 
        h1lv1 = lv1conv(h1lv1, 192, 'layer7-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 192, 64, 'layer7-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #max pool
        h1lv1 = self.maxpool(h1lv1, 'max-3')
        h2lv1 = self.maxpool(h2lv1, 'max-3')
        h2lv2 = self.maxpool(h2lv2, 'max-3')

        #8th layer
        #channel: 384 & 128 
        h1lv1 = lv1conv(h1lv1, 384, 'layer8-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 384, 128, 'layer8-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #9th layer
        #channel: 384 & 128 
        h1lv1 = lv1conv(h1lv1, 384, 'layer9-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 384, 128, 'layer9-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #10th layer
        #channel: 384 & 128 
        h1lv1 = lv1conv(h1lv1, 384, 'layer10-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 384, 128, 'layer10-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #max pool
        h1lv1 = self.maxpool(h1lv1, 'max-4')
        h2lv1 = self.maxpool(h2lv1, 'max-4')
        h2lv2 = self.maxpool(h2lv2, 'max-4')

        #11th layer
        #channel: 384 & 128 
        h1lv1 = lv1conv(h1lv1, 384, 'layer11-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 384, 128, 'layer11-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #12th layer
        #channel: 384 & 128 
        h1lv1 = lv1conv(h1lv1, 384, 'layer12-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 384, 128, 'layer12-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #13th layer
        #channel: 384 & 128 
        h1lv1 = lv1conv(h1lv1, 384, 'layer13-lv1')
        h2lv1, h2lv2 = lv2conv(h1lv1, h2lv2, 384, 128, 'layer13-lv2')
        h2lv1 = tf.add(h1lv1, h2lv1)
        h1lv1 = tf.nn.relu(h1lv1)
        h2lv1 = tf.nn.relu(h2lv1)
        h2lv2 = tf.nn.relu(h2lv2)

        #max pool
        h1lv1 = self.maxpool(h1lv1, 'max-5')
        h2lv1 = self.maxpool(h2lv1, 'max-5')
        h2lv2 = self.maxpool(h2lv2, 'max-5')

        #reshape
        h1lv1 = tf.reshape(h1lv1, [-1, 384])
        h2lv1 = tf.reshape(h2lv1, [-1, 384])
        h2lv2 = tf.reshape(h2lv2, [-1, 128])

        #nested fc
        if self.fc==1:
            h1lv1 = self.lv1fc(h1lv1, 384, 'fc1-lv1')
            h2lv1, h2lv2 = self.lv2fc(h2lv1, h2lv2, 384, 128, 'fc1-lv2')
            h2lv1 = tf.add(h1lv1, h2lv1)
            h1lv1 = tf.nn.relu(h1lv1)
            h2lv1 = tf.nn.relu(h2lv1)
            h2lv2 = tf.nn.relu(h2lv2)

            h1lv1 = self.lv1fc(h1lv1, 384, 'fc2-lv1')
            h2lv1, h2lv2 = self.lv2fc(h2lv1, h2lv2, 384, 128, 'fc2-lv2')
            h2lv1 = tf.add(h1lv1, h2lv1)
            h1lv1 = tf.nn.relu(h1lv1)
            h2lv1 = tf.nn.relu(h2lv1)
            h2lv2 = tf.nn.relu(h2lv2)

            lv1 = h1lv1
            lv2 = tf.concat((h2lv1, h2lv2), 3)
            lv1 = self.lv1fc(lv1, 10, 'fc3-lv1')
            lv2 = self.lv1fc(lv2, 10, 'fc3-lv2')

        #seperated fc
        elif self.fc==2:
            lv1 = h1lv1
            lv2 = tf.concat((h2lv1, h2lv2), 3)

            lv1 = tf.lv1fc(lv1, 384, 'fc1-lv1')
            lv2 = tf.lv1fc(lv2, 512, 'fc1-lv2')
            lv1 = tf.nn.relu(lv1)
            lv2 = tf.nn.relu(lv2)

            lv1 = tf.lv1fc(lv1, 384, 'fc2-lv1')
            lv2 = tf.lv1fc(lv2, 512, 'fc2-lv2')
            lv1 = tf.nn.relu(lv1)
            lv2 = tf.nn.relu(lv2)

            lv1 = tf.lv1fc(lv1, 10, 'fc3-lv1')
            lv2 = tf.lv1fc(lv2, 10, 'fc3-lv2')

        o1 = tf.nn.softmax(lv1)
        o2 = tf.nn.softmax(lv2)

        self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=lv1))
        self.loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=lv2))

        correct_predict1 = tf.equal(tf.argmax(lv1,1), tf.argmax(Y,1))
        correct_predict2 = tf.equal(tf.argmax(lv2,1), tf.argmax(Y,1))
        self.accur1 = tf.reduce_mean(tf.cast(correct_predict1, tf.float32))
        self.accur2 = tf.reduce_mean(tf.cast(correct_predict2, tf.float32))

        t_vars = tf.trainable_variables()
        self.l1_vars = [v for v in t_vars if 'lv-1' in v.name] + [v for v in t_vars if 'input' in
                v.name]
        self.l2_vars = self.l1_vars + [v for v in t_vars if 'lv-2' in v.name]

        # print vars
        print(self.l1_vars)
        print(self.l2_vars)

        self.regularizer1 = self.l2loss(l1_vars)
        self.regularizer2 = self.l2loss(l2_vars)

        self.loss1 += self.regularizer1
        self.loss2 += self.regularizer2

        with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
            self.train_step1 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss1,
                    var_list=self.l1_vars)
            self.train_step2 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss2,
                    var_list=self.l1_vars)


    ### functions ###
    def first_layer(self, input):
        with tf.variable_scope('input-layer'):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)

            out1 = tf.contrib.layers.conv2d(x, 48, [3,3], activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init)
            out2 = tf.contrib.layers.conv2d(x, 16, [3,3], activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init)
        return out1, out2

    def lv1conv(self, input, dim, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)

            out = tf.contrib.layers.conv2d(x, dim, [3,3], activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init)
        return out

    def lv2conv(self, input1, input2, dim1, dim2, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #c_init = tf.truncated_normal_initializer(stddev=5e-2)
            c_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)

            concat_input = tf.concat((input1, input2), 3)

            out1 = tf.contrib.layers.conv2d(input2, dim1, [3,3], activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init, scope='lv1')
            out2 = tf.contrib.layers.conv2d(concat_input, dim2, [3,3], activation_fn=None, 
                    weights_initializer=c_init, biases_initializer=b_init, scope='lv2')
        return out1, out2

    def lv1fc(self, input, dim, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #f_init = tf.truncated_normal_initializer(stddev=5e-2)
            f_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)
            return tf.contrib.layers.fully_connected(input, dim, activation_fn=None,
                    weights_initializer=f_init, biases_initializer=b_init)

    def lv2fc(self, input1, input2, dim1, dim2, scope):
        with tf.variable_scope(scope):
            if not self.is_training:
                tf.get_variable_scope().reuse_variables()
            #f_init = tf.truncated_normal_initializer(stddev=5e-2)
            f_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.0)

            concat_input = tf.concat((input1, input2), 3)

            out1 = tf.contrib.layers.fully_connected(input2, dim1, activation_fn=None,
                    weights_initializer=f_init, biases_initializer=b_init)
            out2 = tf.contrib.layers.fully_connected(concat_input, dim2, activation_fn=None,
                    weights_initializer=f_init, biases_initializer=b_init)
        return out1, out2

    def maxpool(self, x, scope):
        return tf.contrib.layers.max_pool2d(x, [2,2], stride=2, padding='SAME', scope=scope)

    def l2loss(self, var_list):
        regul = tf.nn.l2_loss(var_list[0])
        for v in var_list[1:]:
            regul += tf.nn.l2_loss(v)
        return regul
