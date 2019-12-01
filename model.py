import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
input_dim = 1200
batch_size = 12
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
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        padding="SAME",
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(4e-5), normalizer_fn=slim.batch_norm, 
                        normalizer_params=batch_norm_params,
                        reuse=False):
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
        net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        
        return net

def inception_B(net, prefix='common_layer'):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding='SAME', stride=2,
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
        net = tf.concat([branch_0,branch_1,branch_2],3)

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

def alex_net(inputs):
        net = slim.conv2d()

        
def complete_model(net, is_training, is_inception=True):
        net = encoder_decoder(net)
        net = connected_layer(net)

        if is_inception:
                net = inception_A(net, 'first_layer')
                net = inception_A(net, 'second_layer')
                net = inception_B(net, 'third_layer')
                net = inception_A(net, 'fourth_layer')
                net = inception_A(net, 'fifth_layer')
                net = inception_A(net, 'sixth_layer')
                net = inception_A(net, 'seventh_layer')
        
        pred = classify_model(net, is_training)

        return pred

def train_model(inp, lab, model_dir, dim=input_dim, batch=batch_size, 
                init_lr=0.002, beta=0.2, alpha=2.0, is_training=True):
        
        inp_node = tf.placeholder(shape=[None, 1200, 1200, 1], dtype=tf.float32)
        lab_node = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        
        pred = complete_model(inp_node, is_training)
        print(pred.shape)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=lab_node, logits=pred) 
        loss = tf.reduce_mean(loss)                           
        tf.summary.scalar('losses/Total Loss', loss)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in range(439 // batch_size):
            i = step * batch_size
            j = i + batch_size
            # inp_node: ( batch_size, 1200, 1200, 1 ), lab_node: ( batch_size, 2 )
            _, loss_val = sess.run([optimizer, loss],
                feed_dict = { inp_node: np.random.rand(batch_size,1200,1200, 1), lab_node: np.random.randint(0, 2, (batch_size,2))})
            print(loss_val)
            



# one-hot encoding
#import sklearn
#sklearn.preprocessing.OneHotEncoder().fit_transform(y).toarray()


input_arr = np.random.rand(439,1200,1200)
input_arr[i:j, :, :]
label_arr = np.random.rand(439, 1)
model_dir = './checkpoint/'

train_model(input_arr, label_arr, model_dir, is_training=True)

