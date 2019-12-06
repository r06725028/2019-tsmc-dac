import numpy as np
import os, sys, time
from multiprocessing import  Pool 
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm  
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

forTrain = sys.argv[1]
dirpath = './iccad1/'

if forTrain == 'test':
	print('for test!')
	inpath = dirpath + 'test/'
else:
	inpath = dirpath + 'train/'
outpath = dirpath 

dim_1 = 1200
dim_2 = 1200

def get_label(files):
    if files[0]=='N':
        label=0
    else:
        label=1
    return label

def get_feature(files):
	print("processing files %s"%files)
	#data = plt.imread(os.path.join(inpath, files))
	try:
		data = tf.image.decode_png(tf.read_file(os.path.join(inpath, files)))
		sess = tf.Session()
		data = sess.run(data)
		data = data.astype(np.float32)
		data /= 255 #rescale
		
		#print(files+' ok !!!')
		return data
	except:
		#print('WARNING : '+files)
		return np.array([0])

def mp(filenames):
	pool = Pool(os.cpu_count())
	data_list = []
	data_list.append(pool.map(get_feature, filenames))
	data_list = np.array(data_list[0])
	if forTrain == 'test':
		np.save(outpath+'test_feature.npy', data_list)
	else:
		np.save(outpath+'feature.npy', data_list)
	print(data_list.shape)

	label_list=[]
	label_list.append(pool.map(get_label, filenames))
	label_list = np.array(label_list[0])
	if forTrain == 'test':
		np.save(outpath+'test_label.npy', label_list)
	else:
		np.save(outpath+'label.npy', label_list)
	print(label_list.shape)

def normal(filenames):
	print('begining~~~~')
	data_list = []
	label_list = []
	for files in filenames:
		data = get_feature(files)
		if len(data) == dim_1:
			data_list.append(data)
			label_list.append(get_label(files))
		else:
			with open('./iccad1/false_png.txt', 'a') as f:
				f.write(files)
				f.write('\n')
                
	data_list = np.array(data_list)
	if forTrain == 'test':
		np.save(outpath+'test_feature.npy', data_list)
	else:
		np.save(outpath+'feature.npy', data_list)

	label_list = np.array(label_list)
	if forTrain == 'test':
		np.save(outpath+'test_label.npy', label_list)
	else:
		np.save(outpath+'label.npy', label_list)

	print(data_list.shape)
	print(label_list.shape)
    
	return None

filenames = os.listdir(inpath)
normal(filenames)
