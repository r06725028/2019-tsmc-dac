import os
import tensorflow as tf
import numpy as np

# one-hot encoding
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

# print colorful infos
from termcolor import colored, cprint

from dac_model import DAC

IMG_DIM = 1200
BATCH_SIZE = 32
EPOCH = 5
LEARNING_RATE = 0.002

def magic(string):
    return colored(string, "yellow", "on_grey")

def train_model(train_input, train_label, val_input, val_label, total_size, model_dir, is_training=True):

    train_input_node = tf.placeholder(shape=[None, IMG_DIM, IMG_DIM, 1], dtype=tf.float32)
    train_label_node = tf.placeholder(shape=[None, 2], dtype=tf.int32)
    val_input_node = tf.placeholder(shape=[None, IMG_DIM, IMG_DIM, 1], dtype=tf.float32)
    val_label_node = tf.placeholder(shape=[None, 2], dtype=tf.int32)


    cprint(magic("Initializing DAC model..."))
    model = DAC(is_training)


    # TRAIN PART
    train_pred = model(train_input_node)
    train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_label_node, logits=train_pred)
    train_loss = tf.reduce_mean(train_loss)
    train_correct_prediction = tf.equal(tf.argmax(train_pred, 1), tf.argmax(train_label_node, 1))
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(train_loss)


    # VALIDATION PART
    val_pred = model(val_input_node, use_train_params=True)
    val_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=val_label_node, logits=val_pred)
    val_loss = tf.reduce_mean(val_loss)
    val_correct_prediction = tf.equal(tf.argmax(val_pred, 1), tf.argmax(val_label_node, 1))
    val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))
    
    #tf.summary.scalar('losses/Total Loss', loss)
    
    cprint(magic("Start training/validating model..."))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.Checkpoint(lily_model=model)
        global_step = 0
        for epoch in range(EPOCH):
            for step in range(total_size // BATCH_SIZE):
                i = step * BATCH_SIZE
                j = i + BATCH_SIZE
                # inp_node: ( BATCH_SIZE, IMG_DIM, IMG_DIM, 1 ), lab_node: ( BATCH_SIZE, 2 )
                _, loss_val, acc_val = sess.run([optimizer, train_loss, train_accuracy], 
                        feed_dict = { train_input_node: train_input[i:j, :, :, :], train_label_node: train_label[i:j, :]})
                print('(TRAIN) epoch : {} | step : {} | loss : {} | accuracy : {}'.format(epoch, step, loss_val, acc_val))
                
                global_step += 1
                if global_step % 5 == 0:
                    val_loss_val, val_acc_val = sess.run([val_loss, val_accuracy],
                                                             feed_dict= { val_input_node : val_input, val_label_node: val_label })
                    print('(VALIDATE) epoch : {} | step : {} | loss : {} | accuracy : {}'.format(epoch, step, val_loss_val, val_acc_val))
                    checkpoint.save(os.path.join(model_dir, 'model.cpkt'))
                    print("Model saved in {}".format(model_dir))

if __name__ == '__main__':
        model_dir = './checkpoints/'

        cprint(magic("Load data..."))
        input_arr = np.load('iccad1/feature.npy').reshape(-1,IMG_DIM,IMG_DIM,1)
        label_arr = np.load('iccad1/label.npy').reshape(-1,1)
        cprint(magic("Data loaded."))
        
        # random data for bugs fix convinence
        #input_arr = np.random.rand(439, IMG_DIM, IMG_DIM, 1)
        #label_arr = np.random.randint(low=0, high=2, size=(439, 1))

        label_arr = sklearn.preprocessing.OneHotEncoder().fit_transform(label_arr).toarray()
        cprint(magic("Convert label array into onehot vector..."))

        input_arr, label_arr = shuffle(input_arr, label_arr)

        train_input_arr = input_arr[:400]
        train_label_arr = label_arr[:400]

        # [WARNING] if val size is too large,
        #   you need to make validation process batched, instead of whole.
        val_input_arr = input_arr[400:]
        val_label_arr = label_arr[400:]

        total_size = input_arr.shape[0]

        train_model(train_input_arr, train_label_arr, val_input_arr, val_label_arr, total_size, model_dir, is_training=True)

