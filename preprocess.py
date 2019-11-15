import numpy as np
import os, sys, time
from multiprocessing import  Pool 
import matplotlib.pyplot as plt

dirpath = './iccad1/'
inpath = dirpath + 'train_image/'
outpath = dirpath + 'train_data/'

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
    data = plt.imread(os,path.join(dirname, files))
    data /= 255 #rescale  
    
    return data

tuple_ = os.walk(inpath)   
dirname, dirnames, filenames = tuple_[0], tuple_[1], tuple_[2]

pool = Pool(os.cpu_count())
data_list = []
data_list.append(pool.map(get_feature, filenames))
data_list = np.array(datalist[0])
print(data_list.shape)

label_list=[]
label_list.append(pool.map(get_label, filenames))
label_list = np.array(label_list[0])
print(label_list.shape)

#f.writecsv(outFolder, tmp_data, tmp_label)
