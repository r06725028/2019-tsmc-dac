import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
#from model import complete_model

# one-hot encoding
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

input_dim = 1200
batch_size = 12
inception = True
if inception:
	model_dir = 'incp_' + 'checkpoint/'
	print('This is inception training!')
  
def train_model(input_arr, label_arr, test_inp, test_lab, model_dir, dim=input_dim, batch=batch_size, 
                init_lr=0.002, beta=0.2, alpha=2.0):
	print('training setting......')
	inp_node = tf.placeholder(shape=[12, 1200, 1200, 1], dtype=tf.float32)
	lab_node = tf.placeholder(shape=[12, 2], dtype=tf.int32)
	test_inp_node = tf.placeholder(shape=[843, 1200, 1200, 1], dtype=tf.float32)
	test_lab_node = tf.placeholder(shape=[843, 2], dtype=tf.int32)

	train_pred = complete_model(inp_node, is_training=True, is_inception=inception)
	test_pred = complete_model(test_inp_node, is_training=False, is_inception=inception)
	print('{} shape : {}'.format('training pred', train_pred.shape()))
	print('{} shape : {}'.format('testing pred', test_pred.shape()))

	loss = tf.nn.softmax_cross_entropy_with_logits(labels=lab_node, logits=pred)
	loss = tf.reduce_mean(loss)
	tf.summary.scalar('losses/Total Loss', loss)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
    
	train_true = tf.equal(tf.argmax(train_pred, 1), tf.argmax(lab_node, 1))
	train_acc = tf.reduce_mean(tf.cast(train_true, tf.float32))
	test_true = tf.equal(tf.argmax(test_pred, 1), tf.argmax(test_lab_node, 1))
	test_acc = tf.reduce_mean(tf.cast(test_true, tf.float32))
    
	saver = tf.train.Saver()
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	print('training beginning......')
	for epoch in range(100):
		for step in range(439 // batch_size):
			i = step * batch_size
			j = i + batch_size
			# inp_node: ( batch_size, 1200, 1200, 1 ), lab_node: ( batch_size, 2 )
			_, loss_val, acc_val = sess.run([optimizer, loss, train_acc], 
					feed_dict = { inp_node: input_arr[i:j, :, :, :], lab_node: label_arr[i:j, :]})
			with open(model_dir+'loss.txt', 'a') as f:
				f.write('{}\n'.format(loss_val))
		saver.save(sess, model_dir)
		print('---- epoch:{} | loss:{} | train_acc:{} | test_acc:{}\n'.format(epoch, loss, train_acc, test_acc))
		with open(model_dir+'acc.txt', 'a') as f:
			f.write('---- epoch:{} | loss:{} | train_acc:{} | test_acc:{}\n'.format(epoch, loss, train_acc, test_acc))       

print('load data....')
input_arr = np.load('iccad1/feature.npy').reshape(439,1200,1200,1)
label_arr = np.load('iccad1/label.npy').reshape(439,1)
label_arr = sklearn.preprocessing.OneHotEncoder().fit_transform(label_arr).toarray()
print(label_arr.shape)
input_arr, label_arr = shuffle(input_arr, label_arr)

test_inp = np.load('iccad1/test_feature.npy').reshape(843,1200,1200,1)
test_lab = np.load('iccad1/test_label.npy').reshape(843,1)
test_lab = sklearn.preprocessing.OneHotEncoder().fit_transform(test_lab).toarray()
print(test_lab.shape)

train_model(input_arr, label_arr, test_inp, test_lab, model_dir)
