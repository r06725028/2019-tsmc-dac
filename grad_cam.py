import os
import tensorflow as tf
import numpy as np

# one-hot encoding
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from skimage.transform import resize
from skimage import io

import cv2

from dac_model import DAC


IMG_DIM= 1200
RANDOM_SEED = 1116
CLASS_NUM = 2


def gray2color(img):
    return np.tile(img, [1, 1, 3])


def grad_cam(category_index, layer_name, model_dir, input_arr, sess, class_num=CLASS_NUM):
    """
    Calculate Grad-CAM
      [Code reference]: https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py
      [Explanation]   : https://bindog.github.io/blog/2018/02/10/model-explanation/
    """
    # last layer output for certain class
    loss = tf.multiply(model_output, tf.one_hot([category_index], class_num))
    reduced_loss = tf.reduce_mean(loss)
    # certain layer output
    conv_output = sess.graph.get_tensor_by_name(layer_name + ':0')
    # certain layer gradient
    grads = tf.gradients(reduced_loss, conv_output)[0]
    # normalize
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val = sess.run([conv_output, norm_grads], feed_dict={ inp_node: input_arr })

    output = output[0]
    grads_val = grads_val[0]

    # average weights for each cnn filter
    weights = np.mean(grads_val, axis=(1,2))
    # to sum up R,G,B channes' numbers
    cam = np.ones(output.shape[0 : 2], dtype=np.float32)

    # weighted sum of each cnn filter (filter_size, filter_size, 32) -> (filter_size, filter_size)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]


    # RELU, get rid of negative numbers
    cam = np.maximum(cam, 0)
    # Normalize
    cam = cam / np.max(cam)
    # (filter_size, filter_size) -> (real_img_size, real_img_size)
    cam = resize(cam, (IMG_DIM, IMG_DIM))

    # Converting gray-scale to 3-D
    # (filter_size, filter_size)    -> (filter_size, filter_size, 1)
    cam3 = np.expand_dims(cam, axis=2)
    # (filter_size, filter_size, 1) -> (filter_size, filter_size, 3)
    cam3 = gray2color(cam3)
    return cam3

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES="" python grad_cam.py

    input_arr = np.load('iccad1/feature.npy')[0].reshape(-1, IMG_DIM, IMG_DIM, 1)

    total_size = input_arr.shape[0]

    model_dir = './checkpoint/'

    # get constant output
    tf.set_random_seed(RANDOM_SEED)

    # init model
    model = DAC(is_training=False)
    inp_node = tf.placeholder(shape=[None, IMG_DIM, IMG_DIM, 1], dtype=tf.float32)
    model_output = model(inp_node)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # restore model
        checkpoint = tf.train.Checkpoint(lily_model=model)
        checkpoint.restore(tf.train.latest_checkpoint(os.path.join(model_dir)))

        for target_idx in range(CLASS_NUM):
            cam = grad_cam(target_idx, "dac/dac/conv2_2/Relu", model_dir, input_arr, sess)

            #cam = cv2.applyColorMap(np.uint8(255 * (1 - cam)), cv2.COLORMAP_HSV)  # bigger number, brighter color
            cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HSV)

            raw_img = input_arr[0] # (1, 1200, 1200, 1) -> (1200, 1200, 1)
            raw_img = gray2color(raw_img) # (1200, 1200, 1) -> (1200, 1200, 3)
            raw_img = raw_img * 255 * 255 # 0.00324 -> 255
            raw_img = np.uint8(raw_img) # float32 -> uint8 (unsigned integer 8 bits, 2^8)

            new_img = cv2.addWeighted(cam, 0.7, raw_img, 0.3, 0)

            io.imsave("cam_imgs/new_img_{}.png".format(target_idx), new_img)
            print("save image: cam_imgs/new_img_{}.png".format(target_idx))

