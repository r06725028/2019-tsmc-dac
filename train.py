import os
import tensorflow as tf
import numpy as np

# one-hot encoding
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from dac_model import DAC

IMG_DIM = 1200
BATCH_SIZE = 12

def train_model(input_arr, label_arr, total_size, model_dir, is_training=True):

    inp_node = tf.placeholder(shape=[None, IMG_DIM, IMG_DIM, 1], dtype=tf.float32)
    lab_node = tf.placeholder(shape=[None, 2], dtype=tf.int32)

    model = DAC(is_training)
    pred = model(inp_node)
        
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=lab_node, logits=pred)
    loss = tf.reduce_mean(loss)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(lab_node, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('losses/Total Loss', loss)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    checkpoint = tf.train.Checkpoint(lily_model=model)
    sess.run(init)
    global_step = 0
    for epoch in range(100):
        for step in range(total_size // BATCH_SIZE):
            i = step * BATCH_SIZE
            j = i + BATCH_SIZE
            # inp_node: ( BATCH_SIZE, IMG_DIM, IMG_DIM, 1 ), lab_node: ( BATCH_SIZE, 2 )
            _, loss_val, acc_val = sess.run([optimizer, loss, accuracy], 
                    feed_dict = { inp_node: input_arr[i:j, :, :, :], lab_node: label_arr[i:j, :]})
            print(loss_val)
            global_step += 1
            if global_step % 1000 == 0:
                checkpoint.save(os.path.join(model_dir, ' model.cpkt'))
        print('----'+'epoch :'+str(epoch)+' | accuracy :'+str(acc_val))

if __name__ == '__main__':
        #input_arr = np.load('iccad1/feature.npy').reshape(-1,IMG_DIM,IMG_DIM,1)
        #label_arr = np.load('iccad1/label.npy').reshape(-1,1)
        
        # random gen for bugs fix convinence
        input_arr = np.random.rand(439, IMG_DIM, IMG_DIM, 1)
        label_arr = np.random.randint(low=0, high=2, size=(439, 1))
        label_arr = sklearn.preprocessing.OneHotEncoder().fit_transform(label_arr).toarray()

        input_arr, label_arr = shuffle(input_arr, label_arr)
        model_dir = './checkpoints/'

        train_model(input_arr, label_arr, input_arr.shape[0], model_dir, is_training=True)

