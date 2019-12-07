import tensorflow as tf
import numpy as np
from tqdm import tqdm

# one-hot encoding
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from dac_model import DAC

IMG_DIM= 1200
BATCH_SIZE = 32

def test_model(input_arr, total_size, model_dir,\
                dim=input_dim, is_training=False):

    inp_node = tf.placeholder(shape=[None, IMG_DIM, IMG_DIM, 1], dtype=tf.float32)

    model = DAC(is_training)
    pred_prob = model(inp_node)

    result_arr = []
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # restore model
        checkpoint = tf.train.Checkpoint(lily_model=model)
        # latest checkpoint
        checkpoint.resotre(tf.train.latest_checkpoint(os.path.join(model_dir))
        # specific checkpoint
        # checkpoint.restore(os.path.join(model_dir, 'model.ckpt-1'))
        i = j = 0
        for step in tqdm(range(total_size // BATCH_SIZE)):
            i = step * BATCH_SIZE
            j = i + BATCH_SIZE

            # inp_node: ( total_size, IMG_DIM, IMG_DIM, 1 )
            prob = sess.run(pred_prob, 
                feed_dict = { inp_node: input_arr[i:j, :, :, :] })
            result_arr.append(prob)
            print(str(prob))

        # the last remaining part whose amout is less than BATCH_SIZE
        prob = sess.run(pred_prob,  feed_dict={ inp_node: input_arr[j:, :, :, :] })
        print(str(prob))
        result_arr.append(prob)

    # (batch_num, BATCH_SIZE, 2) -> (total_size, 2)
    result = np.vstack(result_arr)
    return result

if __name__ == '__main__':
        #input_arr = np.load('iccad1/feature.npy').reshape(-1, IMG_DIM, IMG_DIM, 1)
        #label_arr = np.load('iccad1/label.npy').reshape(-1, 1)
        input_arr = np.random.rand(65, IMG_DIM, IMG_DIM, 1)
        label_arr = np.random.randint(low=0, high=2, size=(65, 1))
        label_arr = sklearn.preprocessing.OneHotEncoder().fit_transform(label_arr).toarray()

        model_dir = './checkpoint/'

        # * we set "is_training = False" for model parameters
        pred_dist = test_model(input_arr, total_size=input_arr.shape[0], model_dir=model_dir, is_training=False)

