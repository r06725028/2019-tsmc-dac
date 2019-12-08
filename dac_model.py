import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

batch_norm_var_collection = "moving_vars"
batch_norm_params = {"decay":0.9997, "epsilon":0.001, "updates_collections":tf.GraphKeys.UPDATE_OPS, 
"variables_collections":{"beta":None,"gamma":None, "moving_mean":[batch_norm_var_collection],"moving_variance":[batch_norm_var_collection]}}

def encoder_decoder(net):
    with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu, stride=2, 
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
            weights_regularizer=slim.l2_regularizer(4e-4), normalizer_fn=slim.batch_norm, 
            normalizer_params=batch_norm_params):
        net = slim.conv2d(net, 64, [3, 3], scope='conv1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv2')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3')
        net = slim.conv2d_transpose(net, 256, [3, 3], scope='deconv1')
        net = slim.conv2d_transpose(net, 128, [3, 3], scope='deconv2')
        net = slim.conv2d_transpose(net, 64, [3, 3], scope='deconv3')

    return net

def connected_layer(net):
    net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv4')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
    net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
    net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv6')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')

    return net

def inception_A(net, prefix='common_layer'):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, 
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
            padding="SAME", activation_fn=tf.nn.relu, 
            weights_regularizer=slim.l2_regularizer(4e-5), normalizer_fn=slim.batch_norm, 
            normalizer_params=batch_norm_params):
        with tf.variable_scope("Branch_0"):
            branch_0 = slim.conv2d(net, 64, [1,1], scope=prefix+"conv_0a_1x1")
        with tf.variable_scope("Branch_1"):
            branch_1 = slim.conv2d(net, 48, [1,1], scope=prefix+"conv_1a_1x1")
            branch_1 = slim.conv2d(branch_1, 64, [3,3],scope=prefix+"conv_1a_3X3")
        with tf.variable_scope("Branch_2"):
            branch_2 = slim.conv2d(net, 64, [1,1], scope=prefix+"conv_2a_1x1")
            branch_2 = slim.conv2d(branch_2, 96, [3,3], scope=prefix+"conv_2a_3x3")
            branch_2 = slim.conv2d(branch_2, 96, [3,3], scope=prefix+"conv_2a2_3x3")
        with tf.variable_scope("Branch_3"):

            branch_3 = slim.avg_pool2d(net, [3,3], scope=prefix+"avgpool_3a_3x3", stride=1,padding="SAME")
            branch_3 = slim.conv2d(branch_3, 32, [1,1], scope=prefix+"conv_3a_1x1")

        net = tf.concat([branch_0,branch_1,branch_2,branch_3], 3)

        return net

def inception_B(net, prefix='common_layer'):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME', stride=2, 
            activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
            weights_regularizer=tf.contrib.layers.l2_regularizer(4e-5), normalizer_fn=slim.batch_norm, 
            normalizer_params=batch_norm_params):

        with tf.variable_scope("Branch_0"):
            branch_0 = slim.conv2d(net, 48, [1,1], scope=prefix+"conv_0b_1x1")
            branch_0 = slim.conv2d(branch_0, 64, [3,3],scope=prefix+"conv_0b_3X3", stride=1)
        with tf.variable_scope("Branch_1"):
            branch_1 = slim.conv2d(net, 64, [1,1], scope=prefix+"conv_1b_1x1")
            branch_1 = slim.conv2d(branch_1, 96, [3,3], scope=prefix+"conv_1b_3x3", stride=1)
            branch_1 = slim.conv2d(branch_1, 96, [3,3], scope=prefix+"conv_1b2_3x3", stride=1)
        with tf.variable_scope("Branch_2"):
            branch_2 = slim.avg_pool2d(net, [3,3], scope=prefix+"avgpool_2b_3x3", padding='SAME', stride=2)
            net = tf.concat([branch_0,branch_1,branch_2], 3)

        return net

def classify_model(net, is_training):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME', 
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False), 
            biases_initializer=tf.constant_initializer(0.0)):
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_1')
        net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
        net = slim.flatten(net)
        w_init = tf.contrib.layers.xavier_initializer(uniform=False)
        net = slim.fully_connected(net, 250, activation_fn=tf.nn.relu, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
        predict = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')

    return predict


class DAC(tf.keras.Model):
    def __init__(self, is_training, is_inception=True):
        super().__init__()
        self.is_training = is_training
        self.is_inception = is_inception
        self.inception_A = inception_A
        self.inception_B = inception_B
        self.encoder_decoder = encoder_decoder
        self.connected_layer = connected_layer
        self.classify_model = classify_model

    def call(self, inputs, use_train_params=False):
        with tf.variable_scope('dac', reuse=use_train_params):
            net = self.encoder_decoder(inputs)
            net = self.connected_layer(net)

            if self.is_inception:
                net = self.inception_A(net, 'second_layer')
                net = self.inception_B(net, 'third_layer')
                net = self.inception_A(net, 'fourth_layer')
                net = self.inception_A(net, 'fifth_layer')
                net = self.inception_A(net, 'sixth_layer')
                net = self.inception_A(net, 'seventh_layer')

            pred = self.classify_model(net, self.is_training)
            return pred

