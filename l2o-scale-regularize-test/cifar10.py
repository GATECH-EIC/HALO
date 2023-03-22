"""CIFAR10 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
from matplotlib import pyplot as plt

def load_cifar10(data_path):
    """Loads CIFAR10 dataset.
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    num_train_samples = 50000
    
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')
    
    for i in range(1, 6):
        fpath = os.path.join(data_path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
    
    fpath = os.path.join(data_path, 'test_batch')
    x_test, y_test = load_batch(fpath)
    
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    
    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    
    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)
    x_train = x_train.astype(np.float32) / 255.
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32) / 255.
    y_test = y_test.astype(np.int32)
    
    return (x_train, y_train), (x_test, y_test)

# train, test = load_cifar10('cifar10')
# imgs, labels = train
#
# plt.imshow(imgs[0])
# plt.show()
# print(imgs[0].dtype)
# print(labels[0].dtype)