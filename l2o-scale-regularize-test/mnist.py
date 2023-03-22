"""MNIST handwritten digits dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

def load_mnist(path='mnist/mnist.npz'):
  """Loads the MNIST dataset.
  Arguments:
      path: path where to cache the dataset locally
          (relative to ~/.keras/datasets).
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  License:
      Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
      which is a derivative work from original NIST datasets.
      MNIST dataset is made available under the terms of the
      [Creative Commons Attribution-Share Alike 3.0 license.](
      https://creativecommons.org/licenses/by-sa/3.0/)
  """

  with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    x_train = x_train.astype(np.float32) / 255.
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32) / 255.
    y_test = y_test.astype(np.int32)
    
    return (x_train, y_train), (x_test, y_test)
  
# train, test = load_mnist()
# imgs, labels = train
#
# plt.imshow(imgs[0])
# plt.show()
# print(imgs[0].dtype)
# print(labels[0].dtype)