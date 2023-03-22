# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions to generate or load datasets for supervised learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import math
from pandas import read_csv
import pandas as pd
from scipy import stats
import os
MAX_SEED = 4294967295


class Dataset(namedtuple("Dataset", "data labels")):
  """Helper class for managing a supervised learning dataset.

  Args:
    data: an array of type float32 with N samples, each of which is the set
      of features for that sample. (Shape (N, D_i), where N is the number of
      samples and D_i is the number of features for that sample.)
    labels: an array of type int32 or int64 with N elements, indicating the
      class label for the corresponding set of features in data.
  """
  # Since this is an immutable object, we don't need to reserve slots.
  __slots__ = ()

  @property
  def size(self):
    """Dataset size (number of samples)."""
    return len(self.data)

  def batch_indices(self, num_batches, batch_size, shuffle=True):
    """Creates indices of shuffled minibatches.

    Args:
      num_batches: the number of batches to generate
      batch_size: the size of each batch

    Returns:
      batch_indices: a list of minibatch indices, arranged so that the dataset
          is randomly shuffled.

    Raises:
      ValueError: if the data and labels have different lengths
    """
    if len(self.data) != len(self.labels):
      raise ValueError("Labels and data must have the same number of samples.")

    batch_indices = []

    # Follows logic in mnist.py to ensure we cover the entire dataset.
    index_in_epoch = 0
    dataset_size = len(self.data)
    dataset_indices = np.arange(dataset_size)
    if shuffle:
      np.random.shuffle(dataset_indices)

    for _ in range(num_batches):
      start = index_in_epoch
      index_in_epoch += batch_size
      if index_in_epoch > dataset_size:

        # Finished epoch, reshuffle.
        if shuffle:
          np.random.shuffle(dataset_indices)

        # Start next epoch.
        start = 0
        index_in_epoch = batch_size

      end = index_in_epoch
      batch_indices.append(dataset_indices[start:end].tolist())

    return batch_indices


def noisy_parity_class(n_samples,
                       n_classes=2,
                       n_context_ids=5,
                       noise_prob=0.25,
                       random_seed=None):
  """Returns a randomly generated sparse-to-sparse dataset.

  The label is a parity class of a set of context classes.

  Args:
    n_samples: number of samples (data points)
    n_classes: number of class labels (default: 2)
    n_context_ids: how many classes to take the parity of (default: 5).
    noise_prob: how often to corrupt the label (default: 0.25)
    random_seed: seed used for drawing the random data (default: None)
  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  np.random.seed(random_seed)
  x = np.random.randint(0, n_classes, [n_samples, n_context_ids])
  noise = np.random.binomial(1, noise_prob, [n_samples])
  y = (np.sum(x, 1) + noise) % n_classes
  return Dataset(x.astype("float32"), y.astype("int32"))


def random(n_features, n_samples, n_classes=2, sep=1.0, random_seed=None):
  """Returns a randomly generated classification dataset.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    n_classes: number of class labels (default: 2)
    sep: separation of the two classes, a higher value corresponds to
      an easier classification problem (default: 1.0)
    random_seed: seed used for drawing the random data (default: None)

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  # Generate the problem data.
  x, y = make_classification(n_samples=n_samples,
                             n_features=n_features,
                             n_informative=n_features,
                             n_redundant=0,
                             n_classes=n_classes,
                             class_sep=sep,
                             random_state=random_seed)

  return Dataset(x.astype("float32"), y.astype("int32"))


def random_binary(n_features, n_samples, random_seed=None):
  """Returns a randomly generated dataset of binary values.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    random_seed: seed used for drawing the random data (default: None)

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                 else random_seed)
  np.random.seed(random_seed)

  x = np.random.randint(2, size=(n_samples, n_features))
  y = np.zeros((n_samples, 1))

  return Dataset(x.astype("float32"), y.astype("int32"))


def random_symmetric(n_features, n_samples, random_seed=None):
  """Returns a randomly generated dataset of values and their negatives.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    random_seed: seed used for drawing the random data (default: None)

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                 else random_seed)
  np.random.seed(random_seed)

  x1 = np.random.normal(size=(int(n_samples/2), n_features))
  x = np.concatenate((x1, -x1), axis=0)
  y = np.zeros((n_samples, 1))

  return Dataset(x.astype("float32"), y.astype("int32"))


def random_mlp(n_features, n_samples, random_seed=None, n_layers=6, width=20):
  """Returns a generated output of an MLP with random weights.

  Args:
    n_features: number of features (dependent variables)
    n_samples: number of samples (data points)
    random_seed: seed used for drawing the random data (default: None)
    n_layers: number of layers in random MLP
    width: width of the layers in random MLP

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                 else random_seed)
  np.random.seed(random_seed)

  x = np.random.normal(size=(n_samples, n_features))
  y = x
  n_in = n_features
  scale_factor = np.sqrt(2.) / np.sqrt(n_features)
  for _ in range(n_layers):
    weights = np.random.normal(size=(n_in, width)) * scale_factor
    y = np.dot(y, weights).clip(min=0)
    n_in = width

  y = y[:, 0]
  y[y > 0] = 1

  return Dataset(x.astype("float32"), y.astype("int32"))


def mnist(train=True):
  """Returns mnist loaded data.

  Args:
    train: if true use training data, else use testing data

  Returns:
    dataset: A Dataset namedtuple containing the generated data and labels
  """
  # Generate the problem data.
  path = 'mnist/mnist.npz'
  with np.load(path) as f:
    if train:
      x, y = f['x_train'], f['y_train']
    else:
      x, y = f['x_test'], f['y_test']
  x = x.reshape(-1, 28, 28, 1).astype(np.float32) / 255.
  y = y.astype(np.int32)
  return Dataset(x.astype("float32"), y.astype("int32"))

EMPTY_DATASET = Dataset(np.array([], dtype="float32"),
                        np.array([], dtype="int32"))


def batch_indices(data, labels, num_batches, batch_size, shuffle=True):
  """Creates indices of shuffled minibatches.

  Args:
    data: img data
    labels: label data
    num_batches: the number of batches to generate
    batch_size: the size of each batch

  Returns:
    batch_indices: a list of minibatch indices, arranged so that the dataset
        is randomly shuffled.

  Raises:
    ValueError: if the data and labels have different lengths
  """
  # if len(data) != len(labels):
  if data.shape[0] != labels.shape[0]:
    raise ValueError("Labels and data must have the same number of samples.")

  batch_indices = []

  # Follows logic in mnist.py to ensure we cover the entire dataset.
  index_in_epoch = 0
  dataset_size = len(data)
  dataset_indices = np.arange(dataset_size)
  if shuffle:
    np.random.shuffle(dataset_indices)

  for _ in range(num_batches):
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > dataset_size:
      # Finished epoch, reshuffle.
      if shuffle:
        np.random.shuffle(dataset_indices)

      # Start next epoch.
      start = 0
      index_in_epoch = batch_size

    end = index_in_epoch
    batch_indices.append(dataset_indices[start:end].tolist())

  return batch_indices

def lstm_sin(iterations=100, n_batches=128, n_l=10, noise_scale=0.0, noise_type="Gaussian"):
  n_iterations = iterations * n_batches
  train_x_data = np.zeros([n_iterations, n_l, 1])
  train_y_data = np.zeros([n_iterations, 1])
  for i in range(n_iterations):
    phi = np.random.uniform(0.0, 2 * math.pi)
    omega = np.random.uniform(0.0, math.pi / 2)
    A = np.random.uniform(0.0, 10.0)
    for k in range(n_l):
      if noise_type == "Gaussian":
        train_x_data[i][k][0] = A * math.sin(k * omega + phi) + np.random.normal(scale=noise_scale)
      elif noise_type == "Uniform":
        train_x_data[i][k][0] = A * math.sin(k * omega + phi) + np.random.uniform(high=noise_scale)
      else:
        print("No supporting noise type:", noise_type)
        exit()
    train_y_data[i][0] = A * math.sin(n_l * omega + phi)
  test_x_data = np.zeros([n_batches, n_l, 1])
  test_y_data = np.zeros([n_batches, 1])
  for b in range(n_batches):
    phi = np.random.uniform(0.0, 2 * math.pi)
    omega = np.random.uniform(0.0, math.pi / 2)
    A = np.random.uniform(0.0, 10.0)
    for k in range(n_l):
      if noise_type == "Gaussian":
        train_x_data[i][k][0] = A * math.sin(k * omega + phi) + np.random.normal(scale=noise_scale)
      elif noise_type == "Uniform":
        train_x_data[i][k][0] = A * math.sin(k * omega + phi) + np.random.uniform(high=noise_scale)
      else:
        print("No supporting noise type:", noise_type)
        exit()
    test_y_data[b][0] = A * math.sin(n_l * omega + phi)
  return (train_x_data, train_y_data), (test_x_data, test_y_data)

# load a single file as a numpy array
def load_file(filepath):
  dataframe = read_csv(filepath, header=None, delim_whitespace=True)
  return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
  loaded = list()
  for name in filenames:
    data = load_file(prefix + name)
    loaded.append(data)
  # stack group so that features are the 3rd dimension
  loaded = np.dstack(loaded)
  return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
  filepath = prefix + group + '/Inertial Signals/'
  # load all 9 files as a single array
  filenames = list()
  # total acceleration
  filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
  # body acceleration
  filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
  # body gyroscope
  filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
  # load input data
  X = load_group(filenames, filepath)
  # load class output
  y = load_file(prefix + group + '/y_'+group+'.txt')
  return X, y

def lstm_UCI_HAR_Dataset(iterations=100, n_batches=128, prefix='', shuffle=True):
  # load all train
  train_x_data, train_y_data = load_dataset_group('train', prefix+'UCI_HAR_Dataset/')
  # print(train_x_data.shape, train_y_data.shape)
  # load all test
  test_x_data, test_y_data = load_dataset_group('test', prefix+'UCI_HAR_Dataset/')
  # print(test_x_data.shape, test_y_data.shape)
  # zero-offset class values
  train_y_data = train_y_data - 1
  train_y_data = train_y_data.astype(float)
  test_y_data = test_y_data - 1
  test_y_data = test_y_data.astype(float)
  # preform shuffle to the dataset
  if shuffle:
    num_train_samples = train_x_data.shape[0]
    num_test_samples = test_x_data.shape[0]
    shuffle_train = np.arange(num_train_samples)
    shuffle_test = np.arange(num_test_samples)
    np.random.shuffle(shuffle_train)
    np.random.shuffle(shuffle_test)
    train_x_data = np.take(train_x_data, shuffle_train, axis=0)
    train_y_data = np.take(train_y_data, shuffle_train, axis=0)
    test_x_data = np.take(test_x_data, shuffle_test, axis=0)
    test_y_data = np.take(test_y_data, shuffle_test, axis=0)
  return train_x_data, train_y_data, test_x_data, test_y_data


def lstm_WISDM_HAR_Dataset(shuffle=True):
  COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
  ]
  SEGMENT_TIME_SIZE = 180
  TIME_STEP = 100
  data = pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/WISDM/WISDM_ar_v1.1_raw.txt', header=None, names=COLUMN_NAMES)
  data['z-axis'].replace({';': ''}, regex=True, inplace=True)
  data = data.dropna()

  # SHOW GRAPH FOR JOGGING
  data[data['activity'] == 'Jogging'][['x-axis']][:50].plot(subplots=True, figsize=(16, 12), title='Jogging')
  # plt.xlabel('Timestep')
  # plt.ylabel('X acceleration (dg)')

  # SHOW ACTIVITY GRAPH
  activity_type = data['activity'].value_counts().plot(kind='bar', title='Activity type')
  #plt.show()

  # DATA PREPROCESSING
  data_convoluted = []
  labels = []

  # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
  for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
      x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
      y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
      z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
      data_convoluted.append([x, y, z])

      # Label for a data window is the label that appears most commonly
      label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
      labels.append(label)

  # Convert to numpy
  data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

  # One-hot encoding
  labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
  labels = np.argmax(labels, axis=1).reshape((labels.shape[0], 1))
  print("Convoluted data shape: ", data_convoluted.shape)
  print("Labels shape:", labels.shape)


  # SPLIT INTO TRAINING AND TEST SETS
  X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.25, random_state=13)
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  # print("X train size: ", X_train.shape)
  # print("X test size: ", X_test.shape)
  # print("y train size: ", y_train.shape)
  # print("y test size: ", y_test.shape)

  # print(y_test)

  return X_train, y_train, X_test, y_test


# SML 2010
def window(
    df,
    size,
    driving_series,
    target_series,
):
    X = df[driving_series].values
    y = df[target_series].values
    X_T = []
    y_T = []
    for i in range(len(X) - size + 1):
        X_T.append(X[i : i + size])
        y_T.append(y[i : i + size])

    return np.array(X_T), np.array(y_T)

def get_np_dataset(
    config, cat_before_window=False):
    dfs = []
    for path in config.data_paths:
        dfs.append(pd.read_csv(path, sep=config.sep, usecols=config.usecols))

    df = None
    X_T = None
    y_T = None
    if cat_before_window:
        df = pd.concat(dfs)
        X_T, y_T = window(
            df, config.T, config.driving_series, config.target_cols
        )
        X_T = X_T.transpose((0, 2, 1))
    else:
        X_Ts = []
        y_Ts = []
        for df in dfs:
            X_T, y_T = window(
                df, config.T, config.driving_series, config.target_cols
            )
            # X_T = X_T.transpose((0, 2, 1))
            X_Ts.append(X_T)
            y_Ts.append(np.squeeze(y_T))
        X_T = np.vstack(X_Ts)
        y_T = np.vstack(y_Ts)
    return X_T, y_T

from .config import Config

# config = Config.from_file('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/SML_config.json')
# print(os.path.join(os.getcwd(), "problems/SML_config.json"))
config = Config.from_file(os.path.join(os.getcwd(), "problems/SML_config.json"))

def lstm_SML_Dataset(shuffle=True):
  X, y = get_np_dataset(config)
  # SPLIT INTO TRAINING AND TEST SETS
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
  X_train = np.array(X_train)
  X_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  if shuffle:
    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    shuffle_train = np.arange(num_train_samples)
    shuffle_test = np.arange(num_test_samples)
    np.random.shuffle(shuffle_train)
    np.random.shuffle(shuffle_test)
    X_train = np.take(X_train, shuffle_train, axis=0)
    y_train = np.take(y_train, shuffle_train, axis=0)
    X_test = np.take(X_test, shuffle_test, axis=0)
    y_test = np.take(y_test, shuffle_test, axis=0)

  print("X train size: ", X_train.shape)
  print("X test size: ", X_test.shape)
  print("y train size: ", y_train.shape)
  print("y test size: ", y_test.shape)
  # print(y_test)
  return X_train, y_train, X_test, y_test

from sklearn.preprocessing import MaxAbsScaler
# from keras.utils import np_utils
# Gas Classification
def MLP_GAS_Dataset():
    datatrain1=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch1e.csv')
    datatrain2=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch2e.csv')
    datatrain3=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch3e.csv')
    datatrain4=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch4e.csv')
    datatrain5=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch5e.csv')
    datatrain6=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch6e.csv')
    datatrain7=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch7e.csv')
    datatrain8=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch8e.csv')
    datatrain9=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch9e.csv')
    datatrain10=pd.read_csv('/chaojian_shared_datasets/L2O-Adaptation/l2o-scale-regularize-test/problems/Dataset/Batch10e.csv')

    X1=np.array(datatrain1)
    X2=np.array(datatrain2)
    X3=np.array(datatrain3)
    X4=np.array(datatrain4)
    X5=np.array(datatrain5)
    X6=np.array(datatrain6)
    X7=np.array(datatrain7)
    X8=np.array(datatrain8)
    X9=np.array(datatrain9)
    X10=np.array(datatrain10)

    array_list=[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10]
    sample = np.concatenate([X1, X2])
    lengths = [len(X1), len(X2)]
    datatrain_array=np.vstack(array_list)

    xtrain = datatrain_array[:,1:130]
    ytrain = datatrain_array[:,0]

    max_abs_scaler = MaxAbsScaler()
    xtrain = max_abs_scaler.fit_transform(xtrain)

    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=.1, random_state=1)
    y_train = y_train - 1
    y_test = y_test - 1

    #changing target format
    # y_train = np_utils.to_categorical(y_train)
    # y_test=np_utils.to_categorical(y_test)

    # print("X train size: ", X_train.shape)
    # print("X test size: ", X_test.shape)
    # print("y train size: ", y_train.shape)
    # print("y test size: ", y_test.shape)
    # print(y_test[1:100])

    return X_train, y_train, X_test, y_test