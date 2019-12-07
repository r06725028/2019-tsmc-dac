import tensorflow as tf
import numpy as np

# one-hot encoding
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from test import test_model

IMG_DIM= 1200

def grad_cam(model_output, category_index, layer_name, sess, feed_dict, class_num=2):
    """
        Calculate Grad-CAM
    """
    loss = tf.multiply(model_output, tf.one_hot([category_index], class_num))
    reduced_loss = tf.reduce_sum(loss[0])
    conv_output = sess.graph.get_tensor_by_name(layer_name + ':0')
    # d loss / d conv
    grads = tf.gradients(reduced_loss, conv_output)[0]
    output, grads_val = sess.run([conv_output, grads], feed_dict=feed_dict)
    # avg pooling
    weights = np.mean(grads_val, axis=(1,2))
    # to sum up R,G,B channes' numbers
    cams = np.sum(weights * output, axis=3)
    return cams


if __name__ == '__main__':
        #input_arr = np.load('iccad1/feature.npy').reshape(-1, IMG_DIM, IMG_DIM, 1)
        #label_arr = np.load('iccad1/label.npy').reshape(-1, 1)
        input_arr = np.random.rand(1, IMG_DIM, IMG_DIM, 1)
        label_arr = np.random.randint(low=0, high=2, size=(65, 1))
        inp_node = tf.placeholder(shape=[None, IMG_DIM, IMG_DIM, 1], dtype=tf.float32)

        label_arr = sklearn.preprocessing.OneHotEncoder().fit_transform(label_arr).toarray()

        total_size = input_arr.shape[0]

        model_dir = './checkpoint/'

        # * we set "is_training = False" for model parameters
        _, pred_prob, sess = test_model(input_arr, total_size=input_arr.shape[0], model_dir=model_dir, is_training=False)

        for i in range(total_size):
            item = pred_dist[i]
            for target_idx in range(2):
                grad_cam(pred_prob[i], target_idx, 'later', sess, feed_dict={inp_node:input_arr})

        

