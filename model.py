import numpy as np
import os, sys
import tensorflow.contrib.slim as slim
import tensorflow as tf


def encoder_decoder(inputs):
	with slim.arg_scope([slim.conv2d], padding='VALID', activation_fn=tf.nn.relu, stride=2, 
			weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
			weights_regularizer=slim.l2_regularize(4e-4)):
		net = slim.conv2d(inputs, 64, [3, 3], scope='conv1')
		net = slim.conv2d(net, 128, [3, 3], scope='conv2')
		net = slim.conv2d(net, 256, [3, 3], scope='conv3')
		net = slim.conv2d_transpose(net, 256, [3, 3], scope='deconv1')
		net = slim.conv2d_transpose(net, 128, [3, 3], scope='deconv2')
		net = slim.conv2d_transpose(net, 64, [3, 3], scope='deconv3')
	return neti

def connected_layer(inputs):
	net = slim.conv2d(inputs, 64, [3, 3], stride=1, scope='conv4')
	net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool1')
	net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
	net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv6')
	net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')


	return net

def inception_A(net):
	with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding="SAME", 
			weights_initializer=tf.truncated_normal_initializer(stddev=0.1), activation_fn=tf.nn.relu,
			weights_regularizer=slim.l2_regularize(4e-5), batch_norm_var_collection='moving_vars'):
		with tf.variable_scope("Branch_0"):
			branch_0 = slim.conv2d(net, 64, [1,1], scope="conv_0a_1x1")
		with tf.variable_scope("Branch_1"):
			branch_1 = slim.conv2d(net, 48, [1,1], scope="conv_1a_1x1")
			branch_1 = slim.conv2d(branch_1, 64, [3,3],scope="conv_1a_3X3")
		with tf.variable_scope("Branch_2"):
			branch_2 = slim.conv2d(net, 64, [1,1], scope="conv_2a_1x1")
			branch_2 = slim.conv2d(branch_2, 96, [3,3], scope="conv_2a_3x3")
			branch_2 = slim.conv2d(branch_2, 96, [3,3], scope="conv_2a_3x3")
		with tf.variable_scope("Branch_3"):
			branch_3 = slim.avg_pool2d(net, [3,3], scope="avgpool_3a_3x3")
			branch_3 = slim.conv2d(branch_3, 32, [1,1], scope="conv_3a_1x1")
		net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
	return net

def inception_B(inputs):
	with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding='SAME', 
			activation_fn=tf.nn.relu, stride=2, weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
			weights_regularizer=slim.l2_regularize(4e-5), batch_norm_var_collection='moving_vars'):
		with tf.variable_scope("Branch_0"):
			branch_0 = slim.conv2d(net, 48, [1,1], scope="conv_0b_1x1")
			branch_0 = slim.conv2d(branch_0, 64, [3,3],scope="conv_0b_3X3")
		with tf.variable_scope("Branch_1"):
			branch_2 = slim.conv2d(net, 64, [1,1], scope="conv_1b_1x1")
			branch_2 = slim.conv2d(branch_2, 96, [3,3], scope="conv_1b_3x3")
			branch_2 = slim.conv2d(branch_2, 96, [3,3], scope="conv_1b_3x3")
		with tf.variable_scope("Branch_2"):
			branch_3 = slim.avg_pool2d(net, [3,3], scope="avgpool_2b_3x3")
		net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

	return net

def train_model(inputs):




inputs = np.load('iccad1/train_data/feature.npy')
