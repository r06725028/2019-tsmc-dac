# uncompyle6 version 3.5.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.5 (default, Aug  7 2019, 00:51:29) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-39)]
# Embedded file name: /research/byu2/hyyang/fte/model.py
# Compiled at: 2018-12-20 15:04:01
import os, string, numpy as np
from itertools import islice
import random, csv, pandas as pd
from scipy.fftpack import dct
from time import time
import json
from scipy.misc import *
import skimage.measure as skm, math, multiprocessing as mtp, cvxopt as cvx

def binsearch(x):
    interval = list(range(500))
    return bisect.bisect_left(interval, x * 500)


def quantization(size, val=1):
    return np.empty(size * size, dtype=int).reshape(size, size).fill(val)


def rescale(img):
    return img / 255


def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')


def softmax(y, type=1):
    sum = 0.0
    for i in xrange(0, len(y)):
        sum += math.exp(y[i])

    return math.exp(y[type]) / sum


def subfeature(imgraw, quanti, fealen):
    if not (len(imgraw) == len(quanti) and len(imgraw[:, 0]) == len(quanti[:, 0])):
        print 'ERROR: Image block must have the same size as Quantization matrix.'
        print 'Abort.'
        quit()
    if fealen > len(imgraw) * len(imgraw[:, 0]):
        print 'ERROR: Feature vector length exceeds block size.'
        print 'Abort.'
        quit()
    img = dct2(imgraw)
    size = fealen
    idx = 0
    scaled = np.divide(img, quanti)
    feature = np.zeros(fealen, dtype=np.int)
    for i in range(0, size):
        if idx >= size:
            break
        elif i == 0:
            feature[0] = scaled[(0, 0)]
            idx = idx + 1
        elif i % 2 == 1:
            for j in range(0, i + 1):
                if idx < size:
                    feature[idx] = scaled[(j, i - j)]
                    idx = idx + 1
                else:
                    break

        elif i % 2 == 0:
            for j in range(0, i + 1):
                if idx < size:
                    feature[idx] = scaled[(i - j, j)]
                    idx = idx + 1
                else:
                    break

    return feature


def cutblock(img, block_size, block_dim):
    blockarray = []
    for i in range(0, block_dim):
        blockarray.append([])
        for j in range(0, block_dim):
            blockarray[i].append(img[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size])

    return np.asarray(blockarray)


def feature(img, block_size, block_dim, quanti, fealen):
    img = rescale(img)
    feaarray = np.empty(fealen * block_dim * block_dim).reshape(fealen, block_dim, block_dim)
    blocked = cutblock(img, block_size, block_dim)
    for i in range(0, block_dim):
        for j in range(0, block_dim):
            featemp = subfeature(blocked[(i, j)], quanti, fealen)
            feaarray[:, i, j] = featemp

    return feaarray


def readcsv(target, fealen=32):
    path = target + '/label.csv'
    label = np.genfromtxt(path, delimiter=',')
    feature = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in xrange(0, len(filenames) - 1):
            if i == 0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature.append(featemp)
            else:
                file = '/ac' + str(i) + '.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).as_matrix()
                feature.append(featemp)

    return (
     np.rollaxis(np.asarray(feature), 0, 3)[:, :, 0:fealen], label)


def writecsv(target, data, label, fealen):
    data = data.reshape(len(data), fealen, len(data[0, 0, 0, :]) * len(data[0, 0, :, 0]))
    for i in range(0, fealen):
        if i == 0:
            path = target + '/dc.csv'
            np.savetxt(path, data[:, i, :], fmt='%d', delimiter=',', newline='\n', comments='#')
        else:
            path = target + '/ac' + str(i) + '.csv'
            np.savetxt(path, data[:, i, :], fmt='%d', delimiter=',', newline='\n', comments='#')

    path = target + '/label.csv'
    np.savetxt(path, label, fmt='%d', delimiter=',', newline='\n', comments='#')


def writecsv3(target, data, label, fealen):
    for i in range(0, fealen):
        if i == 0:
            path = target + '/dc.csv'
            np.savetxt(path, data[:, :, i], fmt='%d', delimiter=',', newline='\n', comments='#')
        else:
            path = target + '/ac' + str(i) + '.csv'
            np.savetxt(path, data[:, :, i], fmt='%d', delimiter=',', newline='\n', comments='#')

    path = target + '/label.csv'
    np.savetxt(path, label, fmt='%d', delimiter=',', newline='\n', comments='#')