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

"""Generates toy optimization problems.

This module contains a base class, Problem, that defines a minimal interface
for optimization problems, and a few specific problem types that subclass it.

Test functions for optimization: http://www.sfu.ca/~ssurjano/optimization.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import pdb
from problems import problem_spec as prob_spec

import uuid
import pickle
import itertools

tf.app.flags.DEFINE_float("l2_reg_scale", 1e-3,
                          """Scaling factor for parameter value regularization
                             in softmax classifier problems.""")

FLAGS = tf.app.flags.FLAGS

EPSILON = 1e-6
MAX_SEED = 4294967295
PARAMETER_SCOPE = "parameters"

_Spec = prob_spec.Spec

def expand_list(_len,x):
  assert len(x)<=_len, ("couldn't expand")
  space = int(round(_len/len(x)))
  left = _len-space*len(x)
  expand_x = []
  for i in range(len(x)-1):
    expand_x = expand_x + space*[x[i]]
  expand_x = expand_x + (space+left)*[x[len(x)-1]]
  return expand_x

class Problem(object):
  """Base class for optimization problems.

  This defines an interface for optimization problems, including objective and
  gradients functions and a feed_generator function that yields data to pass to
  feed_dict in tensorflow.

  Subclasses of Problem must (at the minimum) override the objective method,
  which computes the objective/loss/cost to minimize, and specify the desired
  shape of the parameters in a list in the param_shapes attribute.
  """

  def __init__(self, param_shapes, random_seed, noise_stdev, init_fn=None):
    """Initializes a global random seed for the problem.

    Args:
      param_shapes: A list of tuples defining the expected shapes of the
        parameters for this problem
      random_seed: Either an integer (or None, in which case the seed is
        randomly drawn)
      noise_stdev: Strength (standard deviation) of added gradient noise
      init_fn: A function taking a tf.Session object that is used to
        initialize the problem's variables.

    Raises:
      ValueError: If the random_seed is not an integer and not None
    """
    if random_seed is not None and not isinstance(random_seed, int):
      raise ValueError("random_seed must be an integer or None")

    # Pick a random seed.
    self.random_seed = (np.random.randint(MAX_SEED) if random_seed is None
                        else random_seed)

    # Store the noise level.
    self.noise_stdev = noise_stdev

    # Set the random seed to ensure any random data in the problem is the same.
    np.random.seed(self.random_seed)

    # Store the parameter shapes.
    self.param_shapes = param_shapes

    if init_fn is not None:
      self.init_fn = init_fn
    else:
      self.init_fn = lambda _: None

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_normal(shape, seed=seed) for shape in self.param_shapes]

  def init_variables(self, seed=None, pretrained_model_path=None):
    """Returns a list of variables with the given shape."""
    with tf.variable_scope(PARAMETER_SCOPE):
      params = [tf.Variable(param) for param in self.init_tensors(seed, pretrained_model_path=pretrained_model_path)]
    return params

  def objective(self, parameters, data=None, labels=None):
    """Computes the objective given a list of parameters.

    Args:
      parameters: The parameters to optimize (as a list of tensors)
      data: An optional batch of data for calculating objectives
      labels: An optional batch of corresponding labels

    Returns:
      A scalar tensor representing the objective value
    """
    raise NotImplementedError

  def gradients(self, objective, parameters):
    """Compute gradients of the objective with respect to the parameters.

    Args:
      objective: The objective op (e.g. output of self.objective())
      parameters: A list of tensors (the parameters to optimize)

    Returns:
      A list of tensors representing the gradient for each parameter,
        returned in the same order as the given list
    """
    if self.random_sparse_method == "layer_wise" or self.random_sparse_method is None:
      _random_sparse_prob = self.random_sparse_prob
    else:
      _random_sparse_prob = [1.0]

    def real_gradient(p):
      return tf.gradients(objective, parameter)[0]
    def fake_gradient(p):
      return tf.constant(0.0, shape=parameter.shape, dtype=tf.float32)

    parameters_list = list(parameters)
    grads = []
    grad_flag_list = []
    expand_random_sparse_prob = expand_list(len(parameters_list), self.random_sparse_prob)
    assert len(parameters_list) == len(expand_random_sparse_prob), ("Unsuccessful expand")
    for parameter, rd_ratio in zip(parameters_list, expand_random_sparse_prob):
      rd = tf.random.uniform(shape=[], maxval=1)
      grad_flag = tf.math.less_equal(rd, rd_ratio)
      grad_to_append = tf.cond(grad_flag, lambda: real_gradient(parameter), lambda: fake_gradient(parameter))
      grad_flag_list.append(grad_flag)
      grads.append(grad_to_append)

    noisy_grads = []

    for grad in grads:
      if isinstance(grad, tf.IndexedSlices):
        noise = self.noise_stdev * tf.random_normal(tf.shape(grad.values))
        new_grad = tf.IndexedSlices(grad.values + noise, grad.indices)
      else:
        new_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      noisy_grads.append(new_grad)

    return noisy_grads, grad_flag_list

class Quadratic(Problem):
  """Optimizes a random quadratic function.

  The objective is: f(x) = (1/2) ||Wx - y||_2^2
  where W is a random Gaussian matrix and y is a random Gaussian vector.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.0):
    """Initializes a random quadratic problem."""
    param_shapes = [(ndim, 1)]
    super(Quadratic, self).__init__(param_shapes, random_seed, noise_stdev)

    # Generate a random problem instance.
    self.w = np.random.randn(ndim, ndim).astype("float32")
    self.y = np.random.randn(ndim, 1).astype("float32")

  def objective(self, params, data=None, labels=None):
    """Quadratic objective (see base class for details)."""
    return tf.nn.l2_loss(tf.matmul(self.w, params[0]) - self.y)


class SoftmaxClassifier(Problem):
  """Helper functions for supervised softmax classification problems."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_normal(shape, mean=0, stddev=0.01, seed=seed) * 1.2 / np.sqrt(shape[0])
            for shape in self.param_shapes]

  def inference(self, params, data):
    """Computes logits given parameters and data.

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension

    Returns:
      logits: Un-normalized logits with shape (num_samples, num_classes)
    """
    raise NotImplementedError

  def objective(self, params, data, labels, is_training=tf.squeeze(tf.constant([True], dtype=tf.bool))):
    """Computes the softmax cross entropy.

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      loss: Softmax cross entropy loss averaged over the samples in the batch

    Raises:
      ValueError: If the objective is to be computed over >2 classes, because
        this operation is broken in tensorflow at the moment.
    """
    # Forward pass.
    # pdb.set_trace()
    try:
      logits = self.inference(params, data, is_training=is_training)
    except:
      logits = self.inference(params, data)
    # print(logits.get_shape)
    # Compute the loss.
    l2reg = [tf.reduce_sum(param ** 2) for param in params]
    if int(logits.get_shape()[1]) == 2:
      labels = tf.cast(labels, tf.float32)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits[:, 0])
    else:
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
      # from keras.utils import np_utils
      # labels = tf.transpose(labels)
      # # labels = np_utils.to_categorical(labels)
      # losses = tf.nn.softmax_cross_entropy_with_logits(
      #   labels=labels, logits=logits)
      # raise ValueError("Unable to compute softmax cross entropy for more than"
      #                  " 2 classes.")

    return tf.reduce_mean(losses) + tf.reduce_mean(l2reg) * FLAGS.l2_reg_scale

  def argmax(self, logits):
    """Samples the most likely class label given the logits.

    Args:
      logits: Un-normalized logits with shape (num_samples, num_classes)

    Returns:
      predictions: Predicted class labels, has shape (num_samples,)
    """
    return tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)

  def accuracy(self, params, data, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    predictions = self.argmax(self.inference(params, data))
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))

def lstm_func(x, h, c, wx, wh, b):
    """
        x: (N, D)
        h: (N, H)
        c: (N, H)
        wx: (D, 4H)
        wh: (H, 4H)
        b: (4H, )
    """
    N, H = tf.shape(h)[0], tf.shape(h)[1]
    a = tf.reshape(tf.matmul(x, wx) + tf.matmul(h, wh) + b, (N, -1, H))
    i, f, o, g = a[:, 0, :], a[:, 1, :], a[:, 2, :], a[:, 3, :]
    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    g = tf.tanh(g)
    next_c = f * c + i * g
    next_h = o * tf.tanh(next_c)
    return next_h, next_c

class SinLSTMModel(Problem):
  '''A simple sequence prediction task implemented by LSTM.'''
  mnist = None

  def __init__(self, n_batches, n_h, n_l, n_lstm,
               initial_param_scale=0.1, random_seed=None, noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    self.n_h = n_h
    self.n_batches = n_batches
    self.n_l = n_l
    self.n_lstm = n_lstm
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    param_shapes = []
    param_shapes.append((9, 4 * self.n_h))
    # param_shapes.append((4 * self.n_h, ))
    for fltr in range(1, self.n_lstm):
      # Add conv2d filters.
      param_shapes.append((self.n_h, 4 * self.n_h))

    for fltr in range(0, self.n_lstm):
      param_shapes.append((self.n_h, 4 * self.n_h))

    for fltr in range(0, self.n_lstm):
      param_shapes.append((4 * self.n_h, ))

    # Number of units in the final (dense) layer.
    param_shapes.append((self.n_h, 1))  # affine weights
    param_shapes.append((1, ))  # affine bias
    super(SinLSTMModel, self).__init__(param_shapes, random_seed, noise_stdev)

    self.n_batches = n_batches
    self.n_h = n_h
    self.n_l = n_l
    self.n_lstm = n_lstm
    self.initial_param_scale = initial_param_scale
    # self.noise_scale = noise_scale
  def init_tensors(self, seed=None, pretrained_model_path=False):
    """Returns a list of tensors with the given shape."""
    init_params = [tf.random_normal(shape, mean=0., stddev=self.initial_param_scale, seed=seed) for shape in self.param_shapes]
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        init_params[k_id] = k
        print("Loading weight shape:", k.shape)

    return init_params

  def objective(self, parameters, data=None, labels=None):
    lstm_params = parameters[:-2]
    wx = list(lstm_params[:self.n_lstm])
    wh = list(lstm_params[self.n_lstm: 2 * self.n_lstm])
    b = list(lstm_params[2 * self.n_lstm:])
    wo, bo = parameters[-2:]
    h = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]
    c = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]

    for k in range(self.n_l):
      last = data[:, k, :]
      for j in range(self.n_lstm):
        # print(last.shape, h[j].shape, c[j].shape, wx[j].shape, wh[j].shape, b[j].shape)
        h[j], c[j] = lstm_func(last, h[j], c[j], wx[j], wh[j], b[j])
        last = h[j]

    # for calculating accuracy
    predictions = tf.matmul(h[-1], wo) + bo

    return tf.cast(predictions, tf.int32), tf.reduce_mean(tf.square(tf.matmul(h[-1], wo) + bo - labels))

  def argmax(self, logits):
    """Samples the most likely class label given the logits.

    Args:
      logits: Un-normalized logits with shape (num_samples, num_classes)

    Returns:
      predictions: Predicted class labels, has shape (num_samples,)
    """
    return tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)

  def accuracy(self, predictions, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))


class Conv_LSTM(Problem):
  '''A simple sequence prediction task implemented by LSTM.'''
  mnist = None

  def __init__(self, n_batches, n_h, n_l, n_lstm, filter_list, num_classes=1,
               initial_param_scale=0.1, random_seed=None, noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    self.n_h = n_h
    self.n_batches = n_batches
    self.n_l = n_l
    self.n_lstm = n_lstm
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    self.activation=tf.nn.relu
    self.num_classes = num_classes

    param_shapes = []
    # conv
    for fltr in filter_list:
      param_shapes.append((fltr[0], fltr[1], fltr[2]))
      # param_shapes.append((fltr[2],)) # no bias

    self.num_conv = len(filter_list)

    # lstm
    param_shapes.append((1, 4 * self.n_h))
    # param_shapes.append((4 * self.n_h, ))
    for fltr in range(1, self.n_lstm):
      # Add conv2d filters.
      param_shapes.append((self.n_h, 4 * self.n_h))

    for fltr in range(0, self.n_lstm):
      param_shapes.append((self.n_h, 4 * self.n_h))

    for fltr in range(0, self.n_lstm):
      param_shapes.append((4 * self.n_h, ))

    # Number of units in the final (dense) layer.
    param_shapes.append((self.n_h, self.num_classes))  # affine weights
    param_shapes.append((self.num_classes, ))  # affine bias
    super(Conv_LSTM, self).__init__(param_shapes, random_seed, noise_stdev)

    self.n_batches = n_batches
    self.n_h = n_h
    self.n_l = n_l
    self.n_lstm = n_lstm
    self.initial_param_scale = initial_param_scale

    # self.noise_scale = noise_scale

  # New implementation of gradient to limit the work area of sparse updates
  def gradients(self, objective, parameters):
    """Compute gradients of the objective with respect to the parameters.

    Args:
      objective: The objective op (e.g. output of self.objective())
      parameters: A list of tensors (the parameters to optimize)

    Returns:
      A list of tensors representing the gradient for each parameter,
        returned in the same order as the given list
    """
    if self.random_sparse_method == "layer_wise" or self.random_sparse_method is None:
      _random_sparse_prob = self.random_sparse_prob
    else:
      _random_sparse_prob = [1.0]

    def real_gradient(p):
      return tf.gradients(objective, parameter)[0]
    def fake_gradient(p):
      return tf.constant(0.0, shape=parameter.shape, dtype=tf.float32)

    parameters_list = list(parameters)
    grads = []
    grad_flag_list = []
    expand_random_sparse_prob = expand_list(self.num_conv, self.random_sparse_prob)
    assert self.num_conv == len(expand_random_sparse_prob), ("Unsuccessful expand")
    expand_random_sparse_prob = expand_random_sparse_prob + [1.0]*(len(parameters_list)-self.num_conv)
    assert len(parameters_list) == len(expand_random_sparse_prob), ("Unsuccessful expand")
    for parameter, rd_ratio in zip(parameters_list, expand_random_sparse_prob):
      rd = tf.random.uniform(shape=[], maxval=1)
      grad_flag = tf.math.less_equal(rd, rd_ratio)
      grad_to_append = tf.cond(grad_flag, lambda: real_gradient(parameter), lambda: fake_gradient(parameter))
      grad_flag_list.append(grad_flag)
      grads.append(grad_to_append)

    noisy_grads = []

    for grad in grads:
      if isinstance(grad, tf.IndexedSlices):
        noise = self.noise_stdev * tf.random_normal(tf.shape(grad.values))
        new_grad = tf.IndexedSlices(grad.values + noise, grad.indices)
      else:
        new_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      noisy_grads.append(new_grad)

    return noisy_grads, grad_flag_list

  def init_tensors(self, seed=None, pretrained_model_path=False):
    """Returns a list of tensors with the given shape."""
    init_params = [tf.random_normal(shape, mean=0., stddev=self.initial_param_scale, seed=seed) for shape in self.param_shapes]
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        init_params[k_id] = k
        print("Loading weight shape:", k.shape)
        # if k.shape[-1] != 1 or (k_id < len(pretrained_params) - 2):
        #   init_params[k_id] = k
        #   print("Loading weight shape:", k.shape)
        # else:
        #   print("Not loading weight shape:", k.shape)
    return init_params

  def objective(self, parameters, data=None, labels=None):
    # conv weight
    conv_param_num = self.num_conv
    w_conv_list = parameters[:conv_param_num]
    # lstm weight
    lstm_params = parameters[conv_param_num:-2]
    wx = list(lstm_params[:self.n_lstm])
    wh = list(lstm_params[self.n_lstm: 2 * self.n_lstm])
    b = list(lstm_params[2 * self.n_lstm:])
    wo, bo = parameters[-2:]
    h = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]
    c = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]

    # conv1d (128, 128, 9) * (64, 128, 3) | (128, 64, 3) * (64, 64, 3)
    conv_input = tf.transpose(data, perm=[0, 2, 1])
    # print(conv_input.shape)
    # exit()
    for i in range(0, len(w_conv_list)):
      w_conv = w_conv_list[i]
      # b_conv = w_conv_list[i+1]
      # print(conv_input.shape, w_conv.shape)
      layer = tf.nn.conv1d(conv_input, w_conv, stride=3, padding='SAME')
      # layer = tf.nn.bias_add(layer, b_conv) # no bias
      output = self.activation(layer)
      # output = tf.nn.dropout(output, 0.5)
      conv_input = output

    # print(output.shape) # 128, 1, 64
    # exit()

    data = tf.transpose(output, perm=[0, 2, 1])

    # lstm
    # for k in range(self.n_l):
    for k in range(64):
      last = data[:, k, :]
      for j in range(self.n_lstm):
        h[j], c[j] = lstm_func(last, h[j], c[j], wx[j], wh[j], b[j])
        last = h[j]

    # for calculating accuracy
    predictions = tf.matmul(h[-1], wo) + bo

    if self.num_classes == 1:
      return tf.cast(predictions, tf.int32), tf.reduce_mean(tf.square(tf.matmul(h[-1], wo) + bo - labels))
    else:
      return self.argmax(predictions), self.loss(predictions, labels)

  def loss(self, pred, label):
    y_pred_softmax = tf.nn.softmax(pred, name='y_pred_softmax')
    l2 = 0.0015 * sum(tf.nn.l2_loss(i) for i in tf.trainable_variables())
    label = tf.cast(label, tf.int32)
    return  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=tf.one_hot(label, depth=6))) + l2
  def argmax(self, logits):
    """Samples the most likely class label given the logits.

    Args:
      logits: Un-normalized logits with shape (num_samples, num_classes)

    Returns:
      predictions: Predicted class labels, has shape (num_samples,)
    """
    return tf.cast(tf.argmax(logits, 1), tf.int32)

  def accuracy(self, predictions, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))


class Conv_LSTM_v2(Problem):
  '''A simple sequence prediction task implemented by LSTM.'''
  mnist = None

  def __init__(self, n_batches, n_h, filter_list, num_classes=1,
               initial_param_scale=0.1, random_seed=None, noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    self.n_h = n_h
    self.n_batches = n_batches
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    self.activation=tf.nn.relu
    self.num_classes = num_classes

    param_shapes = []
    # conv
    for fltr in filter_list:
      param_shapes.append((fltr[0], fltr[1], fltr[2]))
      # param_shapes.append((fltr[2],)) # no bias

    self.num_conv = len(filter_list)

    param_shapes.append((3, n_h)) # affine weight
    param_shapes.append((n_h, ))  # affine bias

    # lstm
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_h, forget_bias=1.0)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.n_h, forget_bias=1.0)
    self.lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2])

    # Number of units in the final (dense) layer.
    param_shapes.append((self.n_h, self.num_classes))  # affine weights
    param_shapes.append((self.num_classes, ))  # affine bias
    super(Conv_LSTM_v2, self).__init__(param_shapes, random_seed, noise_stdev)

    self.n_batches = n_batches
    self.n_h = n_h
    # self.n_l = n_l
    self.initial_param_scale = initial_param_scale

    # self.noise_scale = noise_scale

  # New implementation of gradient to limit the work area of sparse updates
  def gradients(self, objective, parameters):
    """Compute gradients of the objective with respect to the parameters.

    Args:
      objective: The objective op (e.g. output of self.objective())
      parameters: A list of tensors (the parameters to optimize)

    Returns:
      A list of tensors representing the gradient for each parameter,
        returned in the same order as the given list
    """
    if self.random_sparse_method == "layer_wise" or self.random_sparse_method is None:
      _random_sparse_prob = self.random_sparse_prob
    else:
      _random_sparse_prob = [1.0]

    def real_gradient(p):
      return tf.gradients(objective, parameter)[0]
    def fake_gradient(p):
      return tf.constant(0.0, shape=parameter.shape, dtype=tf.float32)

    parameters_list = list(parameters)
    grads = []
    grad_flag_list = []
    expand_random_sparse_prob = expand_list(len(parameters_list), self.random_sparse_prob)
    assert len(parameters_list) == len(expand_random_sparse_prob), ("Unsuccessful expand")
    for parameter, rd_ratio in zip(parameters_list, expand_random_sparse_prob):
      rd = tf.random.uniform(shape=[], maxval=1)
      grad_flag = tf.math.less_equal(rd, rd_ratio)
      grad_to_append = tf.cond(grad_flag, lambda: real_gradient(parameter), lambda: fake_gradient(parameter))
      grad_flag_list.append(grad_flag)
      grads.append(grad_to_append)

    noisy_grads = []

    for grad in grads:
      if isinstance(grad, tf.IndexedSlices):
        noise = self.noise_stdev * tf.random_normal(tf.shape(grad.values))
        new_grad = tf.IndexedSlices(grad.values + noise, grad.indices)
      else:
        new_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      noisy_grads.append(new_grad)

    return noisy_grads, grad_flag_list

  def init_tensors(self, seed=None, pretrained_model_path=False):
    """Returns a list of tensors with the given shape."""
    init_params = [tf.random_normal(shape, mean=0., stddev=self.initial_param_scale, seed=seed) for shape in self.param_shapes]
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        init_params[k_id] = k
        print("Loading weight shape:", k.shape)
        # if k.shape[-1] != 1 or (k_id < len(pretrained_params) - 2):
        #   init_params[k_id] = k
        #   print("Loading weight shape:", k.shape)
        # else:
        #   print("Not loading weight shape:", k.shape)
    return init_params

  def objective(self, parameters, data=None, labels=None):
    # conv weight
    conv_param_num = self.num_conv
    w_conv_list = parameters[:conv_param_num]
    # fc weight
    wh, bh = parameters[conv_param_num:conv_param_num+2]
    wo, bo = parameters[-2:]

    # conv_input = tf.transpose(data, perm=[0, 2, 1])
    # for i in range(0, len(w_conv_list)):
    #   w_conv = w_conv_list[i]
    #   # b_conv = w_conv_list[i+1]
    #   # print(conv_input.shape, w_conv.shape)
    #   layer = tf.nn.conv1d(conv_input, w_conv, stride=1, padding='SAME')
    #   # layer = tf.nn.bias_add(layer, b_conv) # no bias
    #   output = self.activation(layer)
    #   # output = tf.nn.dropout(output, 0.5)
    #   conv_input = output

    # # print(output.shape) # 128, 1, 64
    # # exit()

    # data = tf.transpose(output, perm=[2, 0, 1])
    data = tf.transpose(data, perm=[1, 0, 2])
    data = tf.reshape(data, [-1, 3])

    # print(data.shape)

    hidden = self.activation(tf.matmul(data, wh) + bh)
    hidden = tf.split(hidden, 180, 0)

    # lstm
    outputs, _ = tf.contrib.rnn.static_rnn(self.lstm_layers, hidden, dtype=tf.float32)
    last_output = outputs[-1]

    # for calculating accuracy
    predictions = tf.matmul(last_output, wo) + bo

    if self.num_classes == 1:
      return tf.cast(predictions, tf.int32), tf.reduce_mean(tf.square(tf.matmul(h[-1], wo) + bo - labels))
    else:
      return self.argmax(predictions), self.loss(predictions, labels)

  def loss(self, pred, label):
    y_pred_softmax = tf.nn.softmax(pred, name='y_pred_softmax')
    l2 = 0.0015 * sum(tf.nn.l2_loss(i) for i in tf.trainable_variables())
    label = tf.one_hot(tf.cast(label, tf.int32), 6)
    print(label)
    return  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)) + l2

  def argmax(self, logits):
    """Samples the most likely class label given the logits.

    Args:
      logits: Un-normalized logits with shape (num_samples, num_classes)

    Returns:
      predictions: Predicted class labels, has shape (num_samples,)
    """
    return tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)

  def accuracy(self, predictions, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))


class Conv_LSTM_v3(Problem):
  '''A simple sequence prediction task implemented by LSTM.'''
  mnist = None

  def __init__(self, n_batches, n_h, n_l, n_lstm, filter_list,
               initial_param_scale=0.1, random_seed=None, noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    self.n_h = n_h
    self.n_batches = n_batches
    self.n_l = n_l
    self.n_lstm = n_lstm
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    self.activation=tf.nn.relu

    param_shapes = []
    # conv
    for fltr in filter_list:
      print(fltr)
      param_shapes.append((fltr[0], fltr[1], fltr[2]))
      # param_shapes.append((fltr[2],)) # no bias

    self.num_conv = len(filter_list)

    # lstm
    param_shapes.append((10, 4 * self.n_h))
    # param_shapes.append((4 * self.n_h, ))
    for fltr in range(1, self.n_lstm):
      # Add conv2d filters.
      param_shapes.append((self.n_h, 4 * self.n_h))

    for fltr in range(0, self.n_lstm):
      param_shapes.append((self.n_h, 4 * self.n_h))

    for fltr in range(0, self.n_lstm):
      param_shapes.append((4 * self.n_h, ))

    # Number of units in the final (dense) layer.
    param_shapes.append((self.n_h, 10))  # affine weights
    param_shapes.append((10, ))  # affine bias
    super(Conv_LSTM_v3, self).__init__(param_shapes, random_seed, noise_stdev)

    self.n_batches = n_batches
    self.n_h = n_h
    self.n_l = n_l
    self.n_lstm = n_lstm
    self.initial_param_scale = initial_param_scale

    # self.noise_scale = noise_scale

  # New implementation of gradient to limit the work area of sparse updates
  def gradients(self, objective, parameters):
    """Compute gradients of the objective with respect to the parameters.

    Args:
      objective: The objective op (e.g. output of self.objective())
      parameters: A list of tensors (the parameters to optimize)

    Returns:
      A list of tensors representing the gradient for each parameter,
        returned in the same order as the given list
    """
    if self.random_sparse_method == "layer_wise" or self.random_sparse_method is None:
      _random_sparse_prob = self.random_sparse_prob
    else:
      _random_sparse_prob = [1.0]

    def real_gradient(p):
      return tf.gradients(objective, parameter)[0]
    def fake_gradient(p):
      return tf.constant(0.0, shape=parameter.shape, dtype=tf.float32)

    parameters_list = list(parameters)
    grads = []
    grad_flag_list = []
    expand_random_sparse_prob = expand_list(self.num_conv, self.random_sparse_prob)
    assert self.num_conv == len(expand_random_sparse_prob), ("Unsuccessful expand")
    expand_random_sparse_prob = expand_random_sparse_prob + [1.0]*(len(parameters_list)-self.num_conv)
    assert len(parameters_list) == len(expand_random_sparse_prob), ("Unsuccessful expand")
    for parameter, rd_ratio in zip(parameters_list, expand_random_sparse_prob):
      rd = tf.random.uniform(shape=[], maxval=1)
      grad_flag = tf.math.less_equal(rd, rd_ratio)
      grad_to_append = tf.cond(grad_flag, lambda: real_gradient(parameter), lambda: fake_gradient(parameter))
      grad_flag_list.append(grad_flag)
      grads.append(grad_to_append)

    noisy_grads = []

    for grad in grads:
      if isinstance(grad, tf.IndexedSlices):
        noise = self.noise_stdev * tf.random_normal(tf.shape(grad.values))
        new_grad = tf.IndexedSlices(grad.values + noise, grad.indices)
      else:
        new_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      noisy_grads.append(new_grad)

    return noisy_grads, grad_flag_list

  def init_tensors(self, seed=None, pretrained_model_path=False):
    """Returns a list of tensors with the given shape."""
    init_params = [tf.random_normal(shape, mean=0., stddev=self.initial_param_scale, seed=seed) for shape in self.param_shapes]
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        init_params[k_id] = k
        print("Loading weight shape:", k.shape)
        # if k.shape[-1] != 1 or (k_id < len(pretrained_params) - 2):
        #   init_params[k_id] = k
        #   print("Loading weight shape:", k.shape)
        # else:
        #   print("Not loading weight shape:", k.shape)
    return init_params

  def objective(self, parameters, data=None, labels=None):
    # conv weight
    conv_param_num = self.num_conv
    w_conv_list = parameters[:conv_param_num]
    # lstm weight
    lstm_params = parameters[conv_param_num:-2]
    wx = list(lstm_params[:self.n_lstm])
    wh = list(lstm_params[self.n_lstm: 2 * self.n_lstm])
    b = list(lstm_params[2 * self.n_lstm:])
    wo, bo = parameters[-2:]
    h = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]
    c = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]

    # conv1d (128, 128, 9) * (64, 128, 3) | (128, 64, 3) * (64, 64, 3)
    # conv_input = tf.transpose(data, perm=[0, 2, 1])
    conv_input = data
    # print(conv_input.shape)
    # exit()
    for i in range(0, len(w_conv_list)):
      w_conv = w_conv_list[i]
      # b_conv = w_conv_list[i+1]
      # print(conv_input.shape, w_conv.shape)
      layer = tf.nn.conv1d(conv_input, w_conv, stride=1, padding='SAME')
      # layer = tf.nn.bias_add(layer, b_conv) # no bias
      output = self.activation(layer)
      # output = tf.nn.dropout(output, 0.5)
      conv_input = output

    # print(output.shape) # 128, 1, 64
    # exit()

    data = tf.transpose(output, perm=[0, 2, 1])
    # data = output

    # lstm
    # for k in range(self.n_l):
    for k in range(32):
      last = data[:, k, :]
      for j in range(self.n_lstm):
        h[j], c[j] = lstm_func(last, h[j], c[j], wx[j], wh[j], b[j])
        last = h[j]

    # for calculating accuracy
    predictions = tf.matmul(h[-1], wo) + bo
    # predictions = h[-1]
    # print(predictions.shape)
    # exit()

    return predictions, self.loss(predictions, labels)

  def loss(self, pred, label):
    return tf.losses.mean_squared_error(pred, label)

  def accuracy(self, predictions, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    return tf.cast(0, tf.float32)


class Conv_UCI(Problem):
  '''A simple sequence prediction task implemented by LSTM.'''
  mnist = None

  def __init__(self, batch_size, filter_list,
               initial_param_scale=0.1, random_seed=None, noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    self.batch_size = batch_size
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    self.activation=tf.nn.relu

    print(filter_list)

    param_shapes = []
    # conv
    for fltr in filter_list:
      param_shapes.append((fltr[0], fltr[1], fltr[2]))

    self.num_conv = len(filter_list)

    # fc
    param_shapes.append((64, 32))  # affine weights
    param_shapes.append((32, ))    # affine bias
    param_shapes.append((32, 16))
    param_shapes.append((16, ))
    param_shapes.append((16, 1))
    param_shapes.append((1, ))

    self.num_fc = 3
    super(Conv_UCI, self).__init__(param_shapes, random_seed, noise_stdev)

    self.initial_param_scale = initial_param_scale
    # self.noise_scale = noise_scale

  # New implementation of gradient to limit the work area of sparse updates
  def gradients(self, objective, parameters):
    """Compute gradients of the objective with respect to the parameters.

    Args:
      objective: The objective op (e.g. output of self.objective())
      parameters: A list of tensors (the parameters to optimize)

    Returns:
      A list of tensors representing the gradient for each parameter,
        returned in the same order as the given list
    """
    if self.random_sparse_method == "layer_wise" or self.random_sparse_method is None:
      _random_sparse_prob = self.random_sparse_prob
    else:
      _random_sparse_prob = [1.0]

    def real_gradient(p):
      return tf.gradients(objective, parameter)[0]
    def fake_gradient(p):
      return tf.constant(0.0, shape=parameter.shape, dtype=tf.float32)

    parameters_list = list(parameters)
    grads = []
    grad_flag_list = []
    expand_random_sparse_prob = expand_list(self.num_conv+self.num_fc*2, self.random_sparse_prob)
    assert self.num_conv+self.num_fc*2 == len(expand_random_sparse_prob), ("Unsuccessful expand")
    for parameter, rd_ratio in zip(parameters_list, expand_random_sparse_prob):
      rd = tf.random.uniform(shape=[], maxval=1)
      grad_flag = tf.math.less_equal(rd, rd_ratio)
      grad_to_append = tf.cond(grad_flag, lambda: real_gradient(parameter), lambda: fake_gradient(parameter))
      grad_flag_list.append(grad_flag)
      grads.append(grad_to_append)

    noisy_grads = []

    for grad in grads:
      if isinstance(grad, tf.IndexedSlices):
        noise = self.noise_stdev * tf.random_normal(tf.shape(grad.values))
        new_grad = tf.IndexedSlices(grad.values + noise, grad.indices)
      else:
        new_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      noisy_grads.append(new_grad)

    return noisy_grads, grad_flag_list

  def init_tensors(self, seed=None, pretrained_model_path=False):
    """Returns a list of tensors with the given shape."""
    init_params = [tf.random_normal(shape, mean=0., stddev=self.initial_param_scale, seed=seed) for shape in self.param_shapes]
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        init_params[k_id] = k
        print("Loading weight shape:", k.shape)
        # if k.shape[-1] != 1 or (k_id < len(pretrained_params) - 2):
        #   init_params[k_id] = k
        #   print("Loading weight shape:", k.shape)
        # else:
        #   print("Not loading weight shape:", k.shape)
    return init_params

  def objective(self, parameters, data=None, labels=None):
    # conv weight
    conv_param_num = self.num_conv
    w_conv_list = parameters[:conv_param_num]
    # fc weight
    fc_list = parameters[conv_param_num:]

    # conv1d (128, 128, 9) * (64, 128, 3) | (128, 64, 3) * (64, 64, 3)
    conv_input = tf.transpose(data, perm=[0, 2, 1])
    for i in range(0, len(w_conv_list)):
      w_conv = w_conv_list[i]
      layer = tf.nn.conv1d(conv_input, w_conv, stride=3, padding='SAME')
      output = self.activation(layer)
      conv_input = output

    # print(output.shape) # 128, 1, 64
    # exit()

    data = tf.squeeze(output)

    # for calculating accuracy
    for i in range(0, len(fc_list), 2):
      w_fc = fc_list[i]
      b_fc = fc_list[i+1]
      predictions = tf.matmul(data, w_fc) + b_fc
      data = predictions

    return tf.cast(predictions, tf.int32), tf.reduce_mean(tf.square(predictions - labels))

  def argmax(self, logits):
    """Samples the most likely class label given the logits.

    Args:
      logits: Un-normalized logits with shape (num_samples, num_classes)

    Returns:
      predictions: Predicted class labels, has shape (num_samples,)
    """
    return tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32)

  def accuracy(self, predictions, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))


class SoftmaxRegression(SoftmaxClassifier):
  """Builds a softmax regression problem."""

  def __init__(self, n_features, n_classes, activation=tf.identity,
               random_seed=None, noise_stdev=0.0):
    self.activation = activation
    self.n_features = n_features
    param_shapes = [(n_features, n_classes), (n_classes,)]
    super(SoftmaxRegression, self).__init__(param_shapes,
                                            random_seed,
                                            noise_stdev)

  def inference(self, params, data):
    features = tf.reshape(data, (-1, self.n_features))
    return tf.matmul(features, params[0]) + params[1]


class SparseSoftmaxRegression(SoftmaxClassifier):
  """Builds a sparse input softmax regression problem."""

  def __init__(self,
               n_features,
               n_classes,
               activation=tf.identity,
               random_seed=None,
               noise_stdev=0.0):
    self.activation = activation
    self.n_features = n_features
    param_shapes = [(n_classes, n_features), (n_features, n_classes), (
        n_classes,)]
    super(SparseSoftmaxRegression, self).__init__(param_shapes, random_seed,
                                                  noise_stdev)

  def inference(self, params, data):
    all_embeddings, softmax_weights, softmax_bias = params
    embeddings = tf.nn.embedding_lookup(all_embeddings, tf.cast(data, tf.int32))
    embeddings = tf.reduce_sum(embeddings, 1)
    return tf.matmul(embeddings, softmax_weights) + softmax_bias


class OneHotSparseSoftmaxRegression(SoftmaxClassifier):
  """Builds a sparse input softmax regression problem.

  This is identical to SparseSoftmaxRegression, but without using embedding
  ops.
  """

  def __init__(self,
               n_features,
               n_classes,
               activation=tf.identity,
               random_seed=None,
               noise_stdev=0.0):
    self.activation = activation
    self.n_features = n_features
    self.n_classes = n_classes
    param_shapes = [(n_classes, n_features), (n_features, n_classes), (
        n_classes,)]
    super(OneHotSparseSoftmaxRegression, self).__init__(param_shapes,
                                                        random_seed,
                                                        noise_stdev)

  def inference(self, params, data):
    all_embeddings, softmax_weights, softmax_bias = params
    num_ids = tf.shape(data)[1]
    one_hot_embeddings = tf.one_hot(tf.cast(data, tf.int32), self.n_classes)
    one_hot_embeddings = tf.reshape(one_hot_embeddings, [-1, self.n_classes])
    embeddings = tf.matmul(one_hot_embeddings, all_embeddings)
    embeddings = tf.reshape(embeddings, [-1, num_ids, self.n_features])
    embeddings = tf.reduce_sum(embeddings, 1)
    return tf.matmul(embeddings, softmax_weights) + softmax_bias


class FullyConnected(SoftmaxClassifier):
  """Builds a multi-layer perceptron classifier."""

  def __init__(self, n_features, n_classes, hidden_sizes=(32, 64),
               activation=tf.nn.sigmoid, random_seed=None, noise_stdev=0.0):
    """Initializes an multi-layer perceptron classification problem."""
    # Store the number of features and activation function.
    self.n_features = n_features
    self.activation = activation

    # Define the network as a list of weight + bias shapes for each layer.
    param_shapes = []
    for ix, sz in enumerate(hidden_sizes + (n_classes,)):

      # The previous layer"s size (n_features if input).
      prev_size = n_features if ix == 0 else hidden_sizes[ix - 1]

      # Weight shape for this layer.
      param_shapes.append((prev_size, sz))

      # Bias shape for this layer.
      param_shapes.append((sz,))

    super(FullyConnected, self).__init__(param_shapes, random_seed, noise_stdev)

  def inference(self, params, data):
    # Flatten the features into a vector.
    features = tf.reshape(data, (-1, self.n_features))

    # Pass the data through the network.
    preactivations = tf.nn.bias_add(tf.matmul(features, params[0]), params[1])

    for layer in range(2, len(self.param_shapes), 2):
      net = self.activation(preactivations)
      preactivations = tf.nn.bias_add(tf.matmul(net, params[layer]), params[layer + 1])
      preactivations = self.activation(preactivations)
    return preactivations

  def accuracy(self, params, data, labels):
    """Computes the accuracy (fraction of correct classifications).

    Args:
      params: List of parameter tensors or variables
      data: Batch of features with samples along the first dimension
      labels: Vector of labels with the same number of samples as the data

    Returns:
      accuracy: Fraction of correct classifications across the batch
    """
    predictions = self.argmax(self.activation(self.inference(params, data)))
    return tf.contrib.metrics.accuracy(predictions, tf.cast(labels, tf.int32))

class MLP(SoftmaxClassifier):
  def __init__(self,
               input_dim,
               n_classes,
               hidden_layer_size_list,
               activation=tf.nn.relu,
               random_seed=None,
               noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):

    # Store the activation.
    self.activation = activation
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    param_shapes = []
    input_size = input_dim
    for hidden_layer_size in hidden_layer_size_list:
      # Add FC layer filters.
      param_shapes.append((input_size, hidden_layer_size))
      param_shapes.append((hidden_layer_size,))
      input_size = hidden_layer_size

    # the final FC before softmax
    param_shapes.append((input_size, n_classes))  # affine weights
    param_shapes.append((n_classes,))  # affine bias

    super(MLP, self).__init__(param_shapes, random_seed, noise_stdev)

  def init_tensors(self, seed=None, pretrained_model_path=None):
    """Returns a list of tensors with the given shape."""
    # load the pretrained model first
    if pretrained_model_path is not None:
      init_params = [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
              for shape in self.param_shapes]
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        # only load before the last FC layer
        if k.shape[-1] != 5 or (k_id < len(pretrained_params) - 2):
          init_params[k_id] = k
          print("Loading weight shape:", k.shape)
        else:
          print("Not loading weight shape:", k.shape)
      return init_params
    else:
      return [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
              for shape in self.param_shapes]

  def inference(self, params, data):

    # Unpack.
    pre_FC_list = params[:-2]
    output_w, output_b = params[-2:]

    MLP_input = data
    # for w_conv in pre_FC_list:
    for i in range(0, len(pre_FC_list), 2):
      w_FC = pre_FC_list[i]
      b_FC = pre_FC_list[i+1]
      output = self.activation(tf.nn.bias_add(tf.matmul(MLP_input, w_FC), b_FC))
      MLP_input = output

    # Fully connected layer.
    # return tf.matmul(flattened, output_w) + output_b
    result = self.activation(tf.nn.bias_add(tf.matmul(MLP_input, output_w), output_b))
    print(result[1:50])
    return result

class ConvNet(SoftmaxClassifier):
  """Builds an N-layer convnet for image classification."""

  def __init__(self,
               image_shape,
               n_classes,
               filter_list,
               activation=tf.nn.relu,
               random_seed=None,
               noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    # Number of channels, number of pixels in x- and y- dimensions.
    n_channels, px, py = image_shape

    # Store the activation.
    self.activation = activation
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    param_shapes = []
    input_size = n_channels
    for fltr in filter_list:
      # Add conv2d filters.
      param_shapes.append((fltr[0], fltr[1], input_size, fltr[2]))
      param_shapes.append((fltr[2],))
      input_size = fltr[2]

    # Number of units in the final (dense) layer.
    self.affine_size = int(input_size * px * py)

    param_shapes.append((self.affine_size, n_classes))  # affine weights
    param_shapes.append((n_classes,))  # affine bias

    super(ConvNet, self).__init__(param_shapes, random_seed, noise_stdev)

  def init_tensors(self, seed=None, pretrained_model_path=None):
    """Returns a list of tensors with the given shape."""
    # load the pretrained model first
    if pretrained_model_path is not None:
      init_params = [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
              for shape in self.param_shapes]
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        # only load before the last FC layer
        if k.shape[-1] != 5 or (k_id < len(pretrained_params) - 2):
          init_params[k_id] = k
          print("Loading weight shape:", k.shape)
        else:
          print("Not loading weight shape:", k.shape)
      return init_params
    else:
      return [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
              for shape in self.param_shapes]

  def inference(self, params, data):

    # Unpack.
    w_conv_list = params[:-2]
    output_w, output_b = params[-2:]

    conv_input = data
    # for w_conv in w_conv_list:
    for i in range(0, len(w_conv_list), 2):
      w_conv = w_conv_list[i]
      b_conv = w_conv_list[i+1]
      layer = tf.nn.conv2d(conv_input, w_conv, strides=[1] * 4, padding="SAME")
      layer = tf.nn.bias_add(layer, b_conv)
      output = self.activation(layer)
      conv_input = output

    # Flatten.
    flattened = tf.reshape(conv_input, (-1, self.affine_size))

    # Fully connected layer.
    # return tf.matmul(flattened, output_w) + output_b
    return self.activation(tf.nn.bias_add(tf.matmul(flattened, output_w), output_b))


class MLP_hy(SoftmaxClassifier):
  """Builds an N-layer convnet for image classification."""

  def __init__(self,
               input_dim,
               n_classes,
               hidden_layer_size_list,
               activation=tf.nn.relu,
               random_seed=None,
               noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    # Number of channels, number of pixels in x- and y- dimensions.
    self.input_size = input_dim

    # Store the activation.
    self.activation = activation
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    param_shapes = []


    param_shapes.append((self.input_size, n_classes))  # affine weights
    param_shapes.append((n_classes,))  # affine bias

    super(MLP_hy, self).__init__(param_shapes, random_seed, noise_stdev)

  def init_tensors(self, seed=None, pretrained_model_path=None):
    """Returns a list of tensors with the given shape."""
    # load the pretrained model first
    if pretrained_model_path is not None:
      init_params = [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
              for shape in self.param_shapes]
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        # only load before the last FC layer
        if k.shape[-1] != 5 or (k_id < len(pretrained_params) - 2):
          init_params[k_id] = k
          print("Loading weight shape:", k.shape)
        else:
          print("Not loading weight shape:", k.shape)
      return init_params
    else:
      return [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
              for shape in self.param_shapes]

  def inference(self, params, data):

    # Unpack.
    w_conv_list = params[:-2]
    output_w, output_b = params[-2:]

    # conv_input = data
    # for w_conv in w_conv_list:
    # for i in range(0, len(w_conv_list), 2):
    #   w_conv = w_conv_list[i]
    #   b_conv = w_conv_list[i+1]
    #   layer = tf.nn.conv2d(conv_input, w_conv, strides=[1] * 4, padding="SAME")
    #   layer = tf.nn.bias_add(layer, b_conv)
    #   output = self.activation(layer)
    #   conv_input = output

    # Flatten.
    # flattened = tf.reshape(conv_input, (-1, self.affine_size))

    # Fully connected layer.
    # return tf.matmul(flattened, output_w) + output_b
    # return self.activation(tf.nn.bias_add(tf.matmul(data, output_w), output_b))
    return tf.nn.bias_add(tf.matmul(data, output_w), output_b)

class ConvNet_pooling(SoftmaxClassifier):
  """Builds an N-layer convnet for image classification."""

  def __init__(self,
               image_shape,
               n_classes,
               filter_list,
               pooling_list,
               activation=tf.nn.relu,
               random_seed=None,
               noise_stdev=0.0):
    # Number of channels, number of pixels in x- and y- dimensions.
    n_channels, px, py = image_shape

    # Store the activation.
    self.activation = activation

    param_shapes = []
    input_size = n_channels
    for fltr in filter_list:
      # Add conv2d filters.
      param_shapes.append((fltr[0], fltr[1], input_size, fltr[2]))
      param_shapes.append((fltr[2],))
      input_size = fltr[2]

    # Number of units in the final (dense) layer.
    self.poolings = pooling_list
    # pdb.set_trace()
    self.affine_size = int(input_size * px * py / 2**len(self.poolings))
    # print(self.affine_size)
    param_shapes.append((self.affine_size, n_classes))  # affine weights
    param_shapes.append((n_classes,))  # affine bias

    super(ConvNet_pooling, self).__init__(param_shapes, random_seed, noise_stdev)

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
            for shape in self.param_shapes]

  def inference(self, params, data):

    # Unpack.
    w_conv_list = params[:-2]
    output_w, output_b = params[-2:]
    # pdb.set_trace()
    conv_input = data
    # for w_conv in w_conv_list:
    for i in range(0, len(w_conv_list), 2):
      # print(conv_input.shape)
      if i in self.poolings:
        conv_input = tf.nn.max_pool(conv_input, ksize=[1, 2, 2, 1],
                                    strides=[1, 1, 1, 1], padding='SAME')
      # print(conv_input.shape)
      w_conv = w_conv_list[i]
      b_conv = w_conv_list[i+1]
      layer = tf.nn.conv2d(conv_input, w_conv, strides=[1] * 4, padding="SAME")
      layer = tf.nn.bias_add(layer, b_conv)
      output = self.activation(layer)
      conv_input = output
    conv_input = tf.nn.max_pool(conv_input, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
    # Flatten.
    flattened = tf.reshape(conv_input, (-1, self.affine_size))

    # Fully connected layer.
    # return tf.matmul(flattened, output_w) + output_b
    return self.activation(tf.nn.bias_add(tf.matmul(flattened, output_w), output_b))

# infact it's a combination of basic block and bottleneck block
def basic_block_shape(in_planes, planes, stride=1):
  block_shape = []
  # Conv 1 without bias
  block_shape.append((1,1,in_planes,planes))
  # BN 1
  block_shape.append((planes,))
  block_shape.append((planes,))
  # Conv 2 without bias
  block_shape.append((3,3,planes,planes))
  # BN 2
  block_shape.append((planes,))
  block_shape.append((planes,))
  # Conv 3 without bias
  block_shape.append((1,1,planes,planes))
  # BN 3
  block_shape.append((planes,))
  block_shape.append((planes,))
  if stride != 1 or in_planes != planes:
    # shortcut Conv with out bias
    block_shape.append((1,1,in_planes,planes))
    # shortcut BN
    block_shape.append((planes,))
    block_shape.append((planes,))
  return block_shape

def PreResNet_Bottleneck_shape(in_planes, planes, cfg, stride=1):
  # Notice: BN running mean/variance, select index are not trainable, so not in the shape
  expansion = 4
  block_shape = []
  # BN 1
  block_shape.append((in_planes,))
  block_shape.append((in_planes,))
  # Conv 1
  block_shape.append((1,1,cfg[0], cfg[1]))
  # BN 2
  block_shape.append((cfg[1],))
  block_shape.append((cfg[1],))
  # Conv 2
  block_shape.append((3,3,cfg[1], cfg[2]))
  # BN 3
  block_shape.append((cfg[2],))
  block_shape.append((cfg[2],))
  # Conv 3
  block_shape.append((1,1,cfg[2], planes * expansion))
  if stride != 1 or in_planes != planes * expansion:
    # downsample Conv
    block_shape.append((1,1,in_planes,planes * expansion))
  return block_shape

class PreResNet(SoftmaxClassifier):
  """Builds an PreResNet with bottleneck block."""
  def __init__(self,
              image_shape,
              n_classes,
              cfg_path,
              select_path,
              depth=20,
              activation=tf.nn.relu,
              random_seed=None,
              noise_stdev=0.0,
              random_sparse_method=None,
              random_sparse_prob=[1.0]):
    expansion = 4
    n_channels, px, py = image_shape
    self.n_channels = n_channels
    # Store the activation.
    self.activation = activation
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    with open(cfg_path, "rb") as cfg_file:
      self.cfg = pickle.load(cfg_file)
    with open(select_path, "rb") as select_file:
      self.select = pickle.load(select_file)
    self.depth = depth

    assert (self.depth - 2) % 9 == 0, 'depth should be 9n+2'
    n = (depth - 2) // 9
    # params shapes, corresponding to ResNet structure
    param_shapes = []
    self.in_planes = n_channels
    # the first Conv params without bias
    param_shapes.append((3,3,self.in_planes,16))
    self.in_planes = 16
    # adding block
    param_shapes.extend(self._make_layer_shape(16, n, cfg = self.cfg[0:3*n]))
    param_shapes.extend(self._make_layer_shape(32, n, cfg = self.cfg[3*n:6*n], stride=2))
    param_shapes.extend(self._make_layer_shape(64, n, cfg = self.cfg[6*n:9*n], stride=2))
    # The BN after the final block
    param_shapes.append((64 * expansion,))
    param_shapes.append((64 * expansion,))
    # adding Linear layer
    param_shapes.append((self.cfg[-1], n_classes)) # for bottleneck block the 512 here should be 512*4
    param_shapes.append((n_classes, ))
    super(PreResNet, self).__init__(param_shapes, random_seed, noise_stdev)

  def _make_layer_shape(self, planes, num_blocks, cfg, stride=1):
    expansion = 4
    layer_shape = []
    layer_shape.extend(PreResNet_Bottleneck_shape(self.in_planes, planes, cfg[0:3], stride))
    self.in_planes = planes*expansion
    for i in range(1, num_blocks):
      layer_shape.extend(PreResNet_Bottleneck_shape(self.in_planes, planes, cfg[3*i: 3*(i+1)]))
    return layer_shape

  def _make_layer_init(self, planes, num_blocks, stride=1):
    expansion = 4
    layer_init = []
    layer_init.extend(self.bottleneck_init(self.init_in_planes, planes, stride))
    self.init_in_planes = planes*expansion
    for i in range(1, num_blocks):
      layer_init.extend(self.bottleneck_init(self.init_in_planes, planes))
    return layer_init

  def bottleneck_init(self, in_planes, planes, stride=1):
    expansion = 4
    block_init = []
    # BN 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 1
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    # BN 2
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 2
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    # BN 3
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 3
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    if stride != 1 or in_planes != planes * expansion:
      # downsample Conv
      shape = self.param_shapes[self.init_index]
      fan_in = float(shape[0]*shape[1]*shape[2])
      block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
      self.init_index += 1
    return block_init
  def init_tensors(self, seed=None, pretrained_model_path=None):
    """Return a list of tensors with the given shape."""
    self.seed = seed
    init_params = []
    self.init_in_planes = self.n_channels
    self.init_index = 0
    # The first Conv
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
    self.init_index += 1
    self.init_in_planes = 16
    # adding block init
    n = (self.depth - 2) // 9
    init_params.extend(self._make_layer_init(16, n))
    init_params.extend(self._make_layer_init(32, n, stride=2))
    init_params.extend(self._make_layer_init(64, n, stride=2))
    # the BN after final block
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Linear layer
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[1])
    init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.zeros(shape))
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        # only load before the last FC layer
        if (k.shape[-1] != 5) or (k_id < len(pretrained_params) - 2):
          init_params[k_id] = k
          print("Loading weight shape:", k.shape)
        else:
          print("Not loading weight shape:", k.shape)
    return init_params

  def _make_layer_imp(self, out, params, planes, num_blocks, cfg, stride=1):
    expansion = 4
    out = self.bottleneck_imp(out, params, self.imp_in_planes, planes, cfg[0:3], stride)
    self.imp_in_planes = planes*expansion
    for i in range(1, num_blocks):
      out = self.bottleneck_imp(out, params, self.imp_in_planes, planes, cfg[3*i: 3*(i+1)])
    return out

  def quantize(self, tensor, bits=8):
    min_th = tf.reduce_min(tensor)
    max_th = tf.reduce_max(tensor)
    width = max_th - min_th

    qrange = 2. ** bits - 1
    scale = tf.div(qrange, width)

    out = tf.clip_by_value(tf.round(tf.multiply(tensor - min_th, scale)), clip_value_min=0,
                           clip_value_max=qrange)
    out = tf.div(out, scale) + min_th

    return out


  def bottleneck_imp(self, _input, params, in_planes, planes, cfg, stride=1):
    expansion = 4
    # BN 1
    out = _input
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    # select
    selected = self.select[self.select_index]
    self.select_index += 1
    selected = np.squeeze(np.argwhere(selected))
    if selected.size == 1:
      selected = np.resize(selected, (1,))
    out = tf.gather(out,selected,axis=3)
    # relu
    out = self.activation(out)
    # Conv 1
    w_conv = params[self.imp_index]
    self.imp_index += 1

    w_conv = self.quantize(w_conv)
    out = self.quantize(out)
    out = tf.nn.conv2d(out, w_conv, strides=[1]*4, padding="SAME")
    # BN 2
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    # relu
    out = self.activation(out)
    # Conv 2
    w_conv = params[self.imp_index]
    self.imp_index += 1

    w_conv = self.quantize(w_conv)
    out = self.quantize(out)
    out = tf.nn.conv2d(out, w_conv, strides=[1, stride, stride, 1], padding="SAME")
    # BN 3
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    # relu
    out = self.activation(out)
    # Conv 3
    w_conv = params[self.imp_index]
    self.imp_index += 1

    w_conv = self.quantize(w_conv)
    out = self.quantize(out)
    out = tf.nn.conv2d(out, w_conv, strides=[1]*4, padding="SAME")
    # downsample
    if stride != 1 or in_planes != planes * expansion:
      # downsample Conv
      w_conv = params[self.imp_index]
      self.imp_index += 1

      w_conv = self.quantize(w_conv)
      _input = self.quantize(_input)
      shortcut = tf.nn.conv2d(_input, w_conv, strides=[1, stride, stride, 1], padding="SAME")
    else:
      shortcut = _input
    out = out + shortcut

    return out

  def inference(self, params, data, is_training=tf.squeeze(tf.constant([True], dtype=tf.bool))):
    # graph = tf.get_default_graph()
    with tf.get_default_graph().as_default() as graph:
      def _clip_grad_op(op, grad):
        x = op.inputs[0]
        x_min = op.inputs[1]
        x_max = op.inputs[2]
        cond = tf.logical_or(tf.less(x, x_min), tf.greater(x, x_max))
        return_grad = tf.where(cond, tf.zeros_like(grad, name="zero_grad"), grad)
        return return_grad, tf.constant(0, name="constant_min_grad"), tf.constant(0, name="constant_max_grad")

      # Register the gradient with a unique id
      grad_name = "MyClipGrad_" + str(uuid.uuid4())
      tf.RegisterGradient(grad_name)(_clip_grad_op)

      with graph.gradient_override_map({"Round": "Identity", "ClipByValue": grad_name}):
        self.is_training = is_training
        conv_input = data
        self.imp_in_planes = self.n_channels
        self.imp_index = 0
        self.select_index = 0 # the index to indicate the current select
        # imp: first Conv
        w_conv = params[self.imp_index]
        self.imp_index += 1
        out = tf.nn.conv2d(conv_input, w_conv, strides=[1]*4, padding="SAME")
        self.imp_in_planes = 16
        # add block implementation
        n = (self.depth - 2) // 9
        out = self._make_layer_imp(out, params, 16, n , cfg = self.cfg[0:3*n])
        out = self._make_layer_imp(out, params, 32, n , cfg = self.cfg[3*n:6*n], stride=2)
        out = self._make_layer_imp(out, params, 64, n , cfg = self.cfg[6*n:9*n], stride=2)
        # add BN after the final block
        gamma_bn = params[self.imp_index]
        self.imp_index += 1
        beta_bn = params[self.imp_index]
        self.imp_index += 1
        out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
        # select here
        selected = self.select[self.select_index]
        self.select_index += 1
        selected = np.squeeze(np.argwhere(selected))
        if selected.size == 1:
          selected = np.resize(selected, (1,))
        out = tf.gather(out,selected,axis=3)
        # relu
        out = self.activation(out)
        # avg_pool
        out = tf.reduce_mean(out, axis=[1,2])
        # FC
        w_fc = params[self.imp_index]
        self.imp_index += 1
        bias_fc = params[self.imp_index]
        self.imp_index += 1
        # flatten
        # print("out before flatten: ", out.shape)
        out = tf.reshape(out, [-1, w_fc.shape[0]])
        out = tf.nn.bias_add(tf.matmul(out, w_fc), bias_fc)
        return out

class ResNet18(SoftmaxClassifier):
  """Builds an ResNet 18."""
  def __init__(self,
              image_shape,
              n_classes,
              num_blocks,
              activation=tf.nn.relu,
              random_seed=None,
              noise_stdev=0.0,
              random_sparse_method=None,
              random_sparse_prob=[1.0]):
    # as I only consider ResNet18 here, so only basicblock is used here
    # Number of channels, number of pixels in x- and y- dimensions.
    n_channels, px, py = image_shape
    self.n_channels = n_channels
    # Store the activation.
    self.activation = activation
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    self.num_blocks = num_blocks
    # params shapes, corresponding to ResNet structure
    param_shapes = []
    self.in_planes = n_channels
    # the first Conv params without bias
    param_shapes.append((3,3,self.in_planes,64))
    self.in_planes = 64
    # BN after the first Conv
    param_shapes.append((self.in_planes,))
    param_shapes.append((self.in_planes,))
    # adding block
    param_shapes.extend(self._make_layer_shape("basic", 64, num_blocks[0], stride=1))
    param_shapes.extend(self._make_layer_shape("basic", 128, num_blocks[1], stride=2))
    param_shapes.extend(self._make_layer_shape("basic", 256, num_blocks[2], stride=2))
    param_shapes.extend(self._make_layer_shape("basic", 512, num_blocks[3], stride=2))
    # adding Linear layer
    param_shapes.append((512, n_classes)) # for bottleneck block the 512 here should be 512*4
    param_shapes.append((n_classes, ))
    super(ResNet18, self).__init__(param_shapes, random_seed, noise_stdev)

  def _make_layer_shape(self, block_type, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layer_shape = []
    for stride in strides:
      if block_type=="basic":
        layer_shape.extend(basic_block_shape(self.in_planes, planes, stride))
        self.in_planes = planes
      elif block_type=="bottleneck":
        print("No implementation for: ", block_type)
        exit()
      else:
        print("No support block type: ", block_type)
        exit()
    return layer_shape

  def _make_layer_init(self, block_type, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layer_init = []
    for stride in strides:
      if block_type=="basic":
        layer_init.extend(self.basic_block_init(self.init_in_planes, planes, stride))
        self.init_in_planes = planes
      elif block_type=="bottleneck":
        print("No implementation for: ", block_type)
        exit()
      else:
        print("No support block type: ", block_type)
        exit()
    return layer_init

  def basic_block_init(self, in_planes, planes, stride=1):
    block_init = []
    # Conv 1 without bias init
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    # BN 1 init
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 2 without bias init
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    # BN 2 init
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 3 without bias init
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    # BN 3 init
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    if stride != 1 or in_planes != planes:
      # shortcut Conv with out bias init
      shape = self.param_shapes[self.init_index]
      fan_in = float(shape[0]*shape[1]*shape[2])
      block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
      self.init_index += 1
      # shortcut BN
      shape = self.param_shapes[self.init_index]
      block_init.append(tf.ones(shape)) # gamma
      self.init_index += 1
      shape = self.param_shapes[self.init_index]
      block_init.append(tf.zeros(shape)) # beta
      self.init_index += 1
    return block_init


  def init_tensors(self, seed=None, pretrained_model_path=None):
    """Return a list of tensors with the given shape."""
    self.seed = seed
    init_params = []
    self.init_in_planes = self.n_channels
    # the first Conv params init
    self.init_index = 0
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
    self.init_index += 1
    self.init_in_planes = 64
    # the BN init
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # adding block init
    init_params.extend(self._make_layer_init("basic", 64, self.num_blocks[0], stride=1))
    init_params.extend(self._make_layer_init("basic", 128, self.num_blocks[1], stride=2))
    init_params.extend(self._make_layer_init("basic", 256, self.num_blocks[2], stride=2))
    init_params.extend(self._make_layer_init("basic", 512, self.num_blocks[3], stride=2))
    # adding Linear layer init
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[1])
    init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.zeros(shape))
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        # only load before the last FC layer
        if (k.shape[-1] != 5) or (k_id < len(pretrained_params) - 2):
          init_params[k_id] = k
          print("Loading weight shape:", k.shape)
        else:
          print("Not loading weight shape:", k.shape)
    return init_params

  def _make_layer_imp(self, _input, params, block_type, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    out = _input
    for stride in strides:
      if block_type=="basic":
        out = self.basic_block_imp(out, params, self.imp_in_planes, planes, stride)
        self.imp_in_planes = planes
      elif block_type=="bottleneck":
        print("No implementation for: ", block_type)
        exit()
      else:
        print("No support block type: ", block_type)
        exit()
    return out

  def basic_block_imp(self, _input, params, in_planes, planes, stride=1):
    # print("====")
    # Conv 1
    w_conv = params[self.imp_index]
    # print("Conv 1: ", w_conv.shape)
    self.imp_index += 1
    out = tf.nn.conv2d(_input, w_conv, strides=[1]*4, padding="SAME")
    # print("Conv 1 out: ", out.shape)
    # BN 1
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    out = self.activation(out)
    # Conv2
    w_conv = params[self.imp_index]
    # print("Conv 2: ", w_conv.shape)
    self.imp_index += 1
    out = tf.nn.conv2d(out, w_conv, strides=[1, stride, stride, 1], padding="SAME")
    # print("Conv 2 out: ", out.shape)
    # BN 2
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    out = self.activation(out)
    # Conv3
    w_conv = params[self.imp_index]
    # print("Conv 3: ", w_conv.shape)
    self.imp_index += 1
    out = tf.nn.conv2d(out, w_conv, strides=[1]*4, padding="SAME")
    # print("Conv 3 out: ", out.shape)
    # BN 3
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))

    if stride != 1 or in_planes != planes:
      # shortcut Conv with out bias imp
      w_conv = params[self.imp_index]
      self.imp_index += 1
      shortcut = tf.nn.conv2d(_input, w_conv, strides=[1, stride, stride, 1], padding="SAME")
      # shortcut BN imp
      gamma_bn = params[self.imp_index]
      self.imp_index += 1
      beta_bn = params[self.imp_index]
      self.imp_index += 1
      shortcut = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, shortcut), lambda: batch_norm_inference(gamma_bn, beta_bn, shortcut))
    else:
      shortcut = _input

    # print(out.shape)
    # print(shortcut.shape)
    # print(in_planes)
    # print(planes)
    out = out + shortcut
    out = self.activation(out)
    # print("final out: ", out.shape)
    # print("====")
    return out

  def inference(self, params, data, is_training=tf.squeeze(tf.constant([True], dtype=tf.bool))):
    self.is_training = is_training
    conv_input = data
    self.imp_in_planes = self.n_channels
    self.imp_index = 0
    # the implementation of the first Conv
    w_conv = params[self.imp_index]
    self.imp_index += 1
    out = tf.nn.conv2d(conv_input, w_conv, strides=[1]*4, padding="SAME")
    self.imp_in_planes =64
    # the implementation of the first BN
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    out = self.activation(out)
    # adding block implementation
    out = self._make_layer_imp(out, params, "basic", 64, self.num_blocks[0], stride=1)
    # print("Layer 1 out: ", out.shape)
    out = self._make_layer_imp(out, params, "basic", 128, self.num_blocks[1], stride=2)
    out = self._make_layer_imp(out, params, "basic", 256, self.num_blocks[2], stride=2)
    out = self._make_layer_imp(out, params, "basic", 512, self.num_blocks[3], stride=2)
    # adding global avg pool
    out = tf.reduce_mean(out, axis=[1,2])

    w_fc = params[self.imp_index]
    self.imp_index += 1
    bias_fc = params[self.imp_index]
    self.imp_index += 1
    # flatten
    # print("out before flatten: ", out.shape)
    out = tf.reshape(out, [-1, w_fc.shape[0]])
    # the last FC layer
    out = tf.nn.bias_add(tf.matmul(out, w_fc), bias_fc)
    return out

def batch_norm_training(gamma, beta, layer):
  zeros = lambda: tf.zeros(gamma.shape)
  ones = lambda: tf.ones(gamma.shape)
  pop_mean = tf.Variable(zeros, trainable=False)
  pop_variance = tf.Variable(ones, trainable=False)
  epsilon = 1e-3
  if len(layer.shape) == 4:
    batch_mean, batch_variance = tf.nn.moments(layer, [0,1,2], keep_dims=False)
  else:
    batch_mean, batch_variance = tf.nn.moments(layer, [0])
  decay = 0.99
  train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
  train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))
  with tf.control_dependencies([train_mean, train_variance]):
    return tf.nn.batch_normalization(layer, batch_mean, batch_variance, beta, gamma, epsilon)

def batch_norm_inference(gamma, beta, layer):
  zeros = lambda: tf.zeros(gamma.shape)
  ones = lambda: tf.ones(gamma.shape)
  pop_mean = tf.Variable(zeros, trainable=False)
  pop_variance = tf.Variable(ones, trainable=False)
  epsilon = 1e-3
  return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)

class vgg16(SoftmaxClassifier):
  """Builds an N-layer convnet for image classification."""

  def __init__(self,
               image_shape,
               n_classes,
               filter_list,
               pooling_list,
               activation=tf.nn.relu,
               random_seed=None,
               noise_stdev=0.0,
               random_sparse_method=None,
               random_sparse_prob=[1.0]):
    # Number of channels, number of pixels in x- and y- dimensions.
    n_channels, px, py = image_shape

    # Store the activation.
    self.activation = activation
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    param_shapes = []
    input_size = n_channels
    for fltr in filter_list:
      # Add conv2d filters.
      param_shapes.append((3, 3, input_size, fltr))
      param_shapes.append((fltr,))
      input_size = fltr

    # Number of units in the final (dense) layer.
    self.poolings = pooling_list
    # pdb.set_trace()
    self.affine_size = int(filter_list[-1])
    # print(self.affine_size)
    param_shapes.append((self.affine_size, n_classes))  # affine weights
    param_shapes.append((n_classes,))  # affine bias

    super(vgg16, self).__init__(param_shapes, random_seed, noise_stdev)

  def init_tensors(self, seed=None, pretrained_model_path=None):
    """Returns a list of tensors with the given shape."""
    # load the pretrained model first
    if pretrained_model_path is not None:
      # init_params = [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
      #         for shape in self.param_shapes]
      init_params = []
      for shape in self.param_shapes:
        if len(shape) == 4:
          fan_in = float(shape[0]*shape[1]*shape[2])
          init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
        elif len(shape) == 2:
          fan_in = float(shape[1])
          init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
        else:
          init_params.append(tf.zeros(shape))
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        # only load before the last FC layer
        if k.shape[-1] != 5 or (k_id < len(pretrained_params) - 2):
          init_params[k_id] = k
          print("Loading weight shape:", k.shape)
        else:
          print("Not loading weight shape:", k.shape)
      return init_params
    else:
      init_params = []
      for shape in self.param_shapes:
        if len(shape) == 4:
          fan_in = float(shape[0]*shape[1]*shape[2])
          init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
        elif len(shape) == 2:
          fan_in = float(shape[1])
          init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
        else:
          init_params.append(tf.zeros(shape))
      return init_params
      # return [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
      #         for shape in self.param_shapes]
    # return [tf.random_normal(shape, mean=0., stddev=0.01, seed=seed)
    #         for shape in self.param_shapes]

  def inference(self, params, data):

    # Unpack.
    w_conv_list = params[:-2]
    output_w, output_b = params[-2:]
    # pdb.set_trace()
    conv_input = data
    # for w_conv in w_conv_list:
    for i in range(0, len(w_conv_list), 2):
      # print(i)
      # print(conv_input.shape)
      if int(i // 2) in self.poolings:
        # print('hhhhh')
        conv_input = tf.nn.max_pool(conv_input, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')
      # print(conv_input.shape)
      w_conv = w_conv_list[i]
      b_conv = w_conv_list[i+1]
      layer = tf.nn.conv2d(conv_input, w_conv, strides=[1] * 4, padding="SAME")
      layer = tf.nn.bias_add(layer, b_conv)
      output = self.activation(layer)
      conv_input = output
    conv_input = tf.nn.max_pool(conv_input, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
    # Flatten.
    flattened = tf.reshape(conv_input, (-1, self.affine_size))

    # Fully connected layer.
    # return tf.matmul(flattened, output_w) + output_b
    return self.activation(tf.nn.bias_add(tf.matmul(flattened, output_w), output_b))

class vgg(SoftmaxClassifier):
  """Builds an N-layer convnet for image classification."""

  def __init__(self,
               image_shape,
               n_classes,
               filter_list,
               activation=tf.nn.relu,
               random_seed=None,
               noise_stdev=0.0,
               padding='SAME'):
    # Number of channels, number of pixels in x- and y- dimensions.
    n_channels, px, py = image_shape

    # Store the activation.
    self.activation = activation
    self.padding=padding
    kernal_size = 3
    param_shapes = ()
    input_size = n_channels
    for fltr in filter_list:
      if fltr == 'M':
        param_shapes.append('M')
      else:
        # Add conv2d filters.
        weights = (kernal_size, kernal_size, input_size, fltr)
        bias = (fltr,)
        param_shapes.append((weights, bias))
        input_size = fltr

    # Number of units in the final (dense) layer.
    self.affine_size = input_size * 1 * 1

    param_shapes.append((self.affine_size, n_classes), (n_classes,))  # affine weights, bias
    # param_shapes.append()  # affine bias

    super(vgg, self).__init__(param_shapes, random_seed, noise_stdev)

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    init_tensors = []
    for shape in self.param_shapes:
      if shape == 'M':
        continue
      else:
        w, b = shape
        weights = tf.random_normal(w, mean=0., stddev=0.01, seed=seed)
        bias = tf.random_normal(b, mean=0., stddev=0.01, seed=seed)
        init_tensors.extend([weights, bias])
    return init_tensors

  def inference(self, params, data):

    # Unpack.
    start = 0
    end = 2
    w_conv_list = params[:-2]
    output_w, output_b = params[-2:]

    conv_input = data
    for shape in self.param_shapes:
      if not isinstance(shape, tuple):
        output = tf.nn.max_pool(conv_input, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding=self.padding)
      else:
        if end > len(w_conv_list):
          raise ValueError('length of the problem paras must stay the same with the maximum index')
        else:
          weights, bias = w_conv_list[start:end]
          layer = tf.nn.conv2d(conv_input, weights, strides=[1] * 4, padding=self.padding)
          layer = tf.nn.bias_add(layer, bias)
          start = end
          end += 2
          output = self.activation(layer)
      conv_input = output

    # Flatten.
    flattened = tf.reshape(conv_input, (-1, self.affine_size))

    # Fully connected layer.
    return self.activation(tf.matmul(flattened, output_w) + output_b)

class Bowl(Problem):
  """A 2D quadratic bowl."""

  def __init__(self, condition_number, angle=0.0,
               random_seed=None, noise_stdev=0.0):
    assert condition_number > 0, "Condition number must be positive."

    # Define parameter shapes.
    param_shapes = [(2, 1)]
    super(Bowl, self).__init__(param_shapes, random_seed, noise_stdev)

    self.condition_number = condition_number
    self.angle = angle
    self._build_matrix(condition_number, angle)

  def _build_matrix(self, condition_number, angle):
    """Builds the Hessian matrix."""
    hessian = np.array([[condition_number, 0.], [0., 1.]], dtype="float32")

    # Build the rotation matrix.
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # The objective is 0.5 * || Ax ||_2^2
    # where the data matrix (A) is: sqrt(Hessian).dot(rotation_matrix).
    self.matrix = np.sqrt(hessian).dot(rotation_matrix)

  def objective(self, params, data=None, labels=None):
    mtx = tf.constant(self.matrix, dtype=tf.float32)
    return tf.nn.l2_loss(tf.matmul(mtx, params[0]))

  def surface(self, xlim=5, ylim=5, n=50):
    xm, ym = _mesh(xlim, ylim, n)
    pts = np.vstack([xm.ravel(), ym.ravel()])
    zm = 0.5 * np.linalg.norm(self.matrix.dot(pts), axis=0) ** 2
    return xm, ym, zm.reshape(n, n)


class Problem2D(Problem):

  def __init__(self, random_seed=None, noise_stdev=0.0):
    param_shapes = [(2,)]
    super(Problem2D, self).__init__(param_shapes, random_seed, noise_stdev)

  def surface(self, n=50, xlim=5, ylim=5):
    """Computes the objective surface over a 2d mesh."""

    # Create a mesh over the given coordinate ranges.
    xm, ym = _mesh(xlim, ylim, n)

    with tf.Graph().as_default(), tf.Session() as sess:

      # Ops to compute the objective at every (x, y) point.
      x = tf.placeholder(tf.float32, shape=xm.shape)
      y = tf.placeholder(tf.float32, shape=ym.shape)
      obj = self.objective([[x, y]])

      # Run the computation.
      zm = sess.run(obj, feed_dict={x: xm, y: ym})

    return xm, ym, zm


class Rosenbrock(Problem2D):
  """See https://en.wikipedia.org/wiki/Rosenbrock_function.

  This function has a single global minima at [1, 1]
  The objective value at this point is zero.
  """

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-5., maxval=10., seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = (1 - x)**2 + 100 * (y - x**2)**2
    return tf.squeeze(obj)


def make_rosenbrock_loss_and_init(device=None):
  """A variable-backed version of Rosenbrock problem.

  See the Rosenbrock class for details.

  Args:
    device: Where to place the ops of this problem.

  Returns:
    A tuple of two callables, first of which creates the loss and the second
    creates the parameter initializer function.
  """
  def make_rosenbrock_loss():
    with tf.name_scope("optimizee"):
      with tf.device(device):
        x = tf.get_variable("x", [1])
        y = tf.get_variable("y", [1])
        c = tf.get_variable(
            "c", [1],
            initializer=tf.constant_initializer(100.0),
            trainable=False)
        obj = (1 - x)**2 + c * (y - x**2)**2
      return tf.squeeze(obj)

  def make_init_fn(parameters):
    with tf.device(device):
      init_op = tf.variables_initializer(parameters)
    def init_fn(sess):
      tf.logging.info("Initializing model parameters.")
      sess.run(init_op)
    return init_fn

  return make_rosenbrock_loss, make_init_fn


class Saddle(Problem2D):
  """Loss surface around a saddle point."""

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = x ** 2 - y ** 2
    return tf.squeeze(obj)


class LogSumExp(Problem2D):
  """2D function defined by the log of the sum of exponentials."""

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = tf.log(tf.exp(x + 3. * y - 0.1) +
                 tf.exp(x - 3. * y - 0.1) +
                 tf.exp(-x - 0.1) + 1.0)
    return tf.squeeze(obj)


class Ackley(Problem2D):
  """Ackley's function (contains many local minima)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-32.768, maxval=32.768, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = (-20 * tf.exp(-0.2 * tf.sqrt(0.5 * (x ** 2 + y ** 2))) -
           tf.exp(0.5 * (tf.cos(2 * np.pi * x) + tf.cos(2 * np.pi * y))) +
           tf.exp(1.0) + 20.)
    return tf.squeeze(obj)


class Beale(Problem2D):
  """Beale function (a multimodal function with sharp peaks)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-4.5, maxval=4.5, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = ((1.5 - x + x * y) ** 2 +
           (2.25 - x + x * y ** 2) ** 2 +
           (2.625 - x + x * y ** 3) ** 2)
    return tf.squeeze(obj)


class Booth(Problem2D):
  """Booth's function (has a long valley along one dimension)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-10., maxval=10., seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    return tf.squeeze(obj)


class StyblinskiTang(Problem2D):
  """Styblinski-Tang function (a bumpy function in two dimensions)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-5., maxval=5., seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    params = tf.split(params[0], 2, axis=0)
    obj = 0.5 * tf.reduce_sum([x ** 4 - 16 * x ** 2 + 5 * x
                               for x in params], 0) + 80.
    return tf.squeeze(obj)


class Matyas(Problem2D):
  """Matyas function (a function with a single global minimum in a valley)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=-10, maxval=10, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    obj = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return tf.squeeze(obj)


class Branin(Problem2D):
  """Branin function (a function with three global minima)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    x1 = tf.random_uniform((1,), minval=-5., maxval=10.,
                           seed=seed)
    x2 = tf.random_uniform((1,), minval=0., maxval=15.,
                           seed=seed)
    return [tf.concat([x1, x2], 0)]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)

    # Define some constants.
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5 / np.pi
    r = 6.
    s = 10.
    t = 1 / (8. * np.pi)

    # Evaluate the function.
    obj = a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * tf.cos(x) + s
    return tf.squeeze(obj)


class Michalewicz(Problem2D):
  """Michalewicz function (has steep ridges and valleys)."""

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return [tf.random_uniform(shape, minval=0., maxval=np.pi, seed=seed)
            for shape in self.param_shapes]

  def objective(self, params, data=None, labels=None):
    x, y = tf.split(params[0], 2, axis=0)
    m = 5    # Defines how steep the ridges are (larger m => steeper ridges).
    obj = 2. - (tf.sin(x) * tf.sin(x ** 2 / np.pi) ** (2 * m) +
                tf.sin(y) * tf.sin(2 * y ** 2 / np.pi) ** (2 * m))
    return tf.squeeze(obj)


class Rescale(Problem):
  """Takes an existing problem, and rescales all the parameters."""

  def __init__(self, problem_spec, scale=10., noise_stdev=0.0):
    self.problem = problem_spec.build()
    self.param_shapes = self.problem.param_shapes
    self.scale = scale

    super(Rescale, self).__init__(self.param_shapes, random_seed=None,
                                  noise_stdev=noise_stdev)

  def init_tensors(self, seed=None):
    params_raw = self.problem.init_tensors(seed=seed)
    params = [t * self.scale for t in params_raw]
    return params

  def objective(self, params, data=None, labels=None):
    params_raw = [t/self.scale for t in params]

    problem_obj = self.problem.objective(params_raw, data, labels)
    return problem_obj


class SumTask(Problem):
  """Takes a list of problems and modifies the objective to be their sum."""

  def __init__(self, problem_specs, noise_stdev=0.0):
    self.problems = [ps.build() for ps in problem_specs]
    self.param_shapes = []
    for prob in self.problems:
      self.param_shapes += prob.param_shapes

    super(SumTask, self).__init__(self.param_shapes, random_seed=None,
                                  noise_stdev=noise_stdev)

  def init_tensors(self, seed=None):
    tensors = []
    for prob in self.problems:
      tensors += prob.init_tensors(seed=seed)
    return tensors

  def objective(self, params, data=None, labels=None):
    obj = 0.
    index = 0
    for prob in self.problems:
      num_params = len(prob.param_shapes)
      obj += prob.objective(params[index:index + num_params])
      index += num_params
    return obj


class IsotropicQuadratic(Problem):
  """An isotropic quadratic problem."""

  def objective(self, params, data=None, labels=None):
    return sum([tf.reduce_sum(param ** 2) for param in params])


class Norm(Problem):
  """Takes an existing problem and modifies the objective to be its N-norm."""

  def __init__(self, ndim, random_seed=None, noise_stdev=0.0, norm_power=2.):
    param_shapes = [(ndim, 1)]
    super(Norm, self).__init__(param_shapes, random_seed, noise_stdev)

    # Generate a random problem instance.
    self.w = np.random.randn(ndim, ndim).astype("float32")
    self.y = np.random.randn(ndim, 1).astype("float32")
    self.norm_power = norm_power

  def objective(self, params, data=None, labels=None):
    diff = tf.matmul(self.w, params[0]) - self.y
    exp = 1. / self.norm_power
    loss = tf.reduce_sum((tf.abs(diff) + EPSILON) ** self.norm_power) ** exp
    return loss


class LogObjective(Problem):
  """Takes an existing problem and modifies the objective to be its log."""

  def __init__(self, problem_spec):
    self.problem = problem_spec.build()
    self.param_shapes = self.problem.param_shapes

    super(LogObjective, self).__init__(self.param_shapes,
                                       random_seed=None,
                                       noise_stdev=0.0)

  def objective(self, params, data=None, labels=None):
    problem_obj = self.problem.objective(params, data, labels)
    return tf.log(problem_obj + EPSILON) - tf.log(EPSILON)


class SparseProblem(Problem):
  """Takes a problem and sets gradients to 0 with the given probability."""

  def __init__(self,
               problem_spec,
               zero_probability=0.99,
               random_seed=None,
               noise_stdev=0.0):
    self.problem = problem_spec.build()
    self.param_shapes = self.problem.param_shapes
    self.zero_prob = zero_probability

    super(SparseProblem, self).__init__(self.param_shapes,
                                        random_seed=random_seed,
                                        noise_stdev=noise_stdev)

  def objective(self, parameters, data=None, labels=None):
    return self.problem.objective(parameters, data, labels)

  def gradients(self, objective, parameters):
    grads = tf.gradients(objective, list(parameters))

    new_grads = []
    for grad in grads:
      mask = tf.greater(self.zero_prob, tf.random_uniform(grad.get_shape()))
      zero_grad = tf.zeros_like(grad, dtype=tf.float32)
      noisy_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      new_grads.append(tf.where(mask, zero_grad, noisy_grad))
    return new_grads


class DependencyChain(Problem):
  """A problem in which parameters must be optimized in order.

  A sequence of parameters which all need to be brought to 0, but where each
  parameter in the sequence can't be brought to 0 until the preceding one
  has been. This should take a long time to optimize, with steady
  (or accelerating) progress throughout the entire process.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(ndim + 1,)]
    self.ndim = ndim
    super(DependencyChain, self).__init__(
        param_shapes, random_seed, noise_stdev)

  def objective(self, params, data=None, labels=None):
    terms = params[0][0]**2 + params[0][1:]**2 / (params[0][:-1]**2 + EPSILON)
    return tf.reduce_sum(terms)


class MinMaxWell(Problem):
  """Problem with global min when both the min and max (absolute) params are 1.

  The gradient for all but two parameters (the min and max) is zero. This
  should therefore encourage the optimizer to behave sensible even when
  parameters have zero gradients, as is common eg for some deep neural nets.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(ndim,)]
    self.ndim = ndim
    super(MinMaxWell, self).__init__(param_shapes, random_seed, noise_stdev)

  def objective(self, params, data=None, labels=None):
    params_sqr = params[0]**2
    min_sqr = tf.reduce_min(params_sqr)
    max_sqr = tf.reduce_max(params_sqr)
    epsilon = 1e-12

    return max_sqr + 1./min_sqr - 2. + epsilon


class OutwardSnake(Problem):
  """A winding path out to infinity.

  Ideal step length stays constant along the entire path.
  """

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(ndim,)]
    self.ndim = ndim
    super(OutwardSnake, self).__init__(param_shapes, random_seed, noise_stdev)

  def objective(self, params, data, labels=None):
    radius = tf.sqrt(tf.reduce_sum(params[0]**2))
    rad_loss = tf.reduce_sum(1. / (radius + 1e-6) * data[:, 0])

    sin_dist = params[0][1:] - tf.cos(params[0][:-1]) * np.pi
    sin_loss = tf.reduce_sum((sin_dist * data[:, 1:])**2)

    return rad_loss + sin_loss


class ProjectionQuadratic(Problem):
  """Dataset consists of different directions to probe. Global min is at 0."""

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(1, ndim)]
    super(ProjectionQuadratic, self).__init__(
        param_shapes, random_seed, noise_stdev)

  def objective(self, params, data, labels=None):
    return tf.reduce_sum((params[0] * data)**2)


class SumOfQuadratics(Problem):

  def __init__(self, ndim, random_seed=None, noise_stdev=0.):
    param_shapes = [(1, ndim)]
    super(SumOfQuadratics, self).__init__(
        param_shapes, random_seed, noise_stdev)

  def objective(self, params, data, labels=None):
    epsilon = 1e-12
    # Assume dataset is designed so that the global minimum is at params=0.
    # Subtract loss at params=0, so that global minimum has objective value
    # epsilon (added to avoid floating point issues).
    return (tf.reduce_sum((params[0] - data)**2) - tf.reduce_sum(data**2) +
            epsilon)


class MatMulAlgorithm(Problem):
  """A 6-th order polynomial optimization problem.

  This problem is parametrized by n and k. A solution to this problem with
  objective value exactly zero defines a matrix multiplication algorithm of
  n x n matrices using k multiplications between matrices. When applied
  recursively, such an algorithm has complexity O(n^(log_n(k))).

  Given n, it is not known in general which values of k in [n^2, n^3] have a
  solution. There is always a solution with k = n^3 (this is the naive
  algorithm).

  In the special case n = 2, it is known that there are solutions for k = {7, 8}
  but not for k <= 6. For n = 3, it is known that there are exact solutions for
  23 <= k <= 27, and there are asymptotic solutions for k = {21, 22}, but the
  other cases are unknown.

  For a given n and k, if one solution exists then infinitely many solutions
  exist due to permutation and scaling symmetries in the parameters.

  This is a very hard problem for some values of n and k (e.g. n = 3, k = 21),
  but very easy for other values (e.g. n = 2, k = 7).

  For a given n and k, the specific formulation of this problem is as follows.
  Let theta_a, theta_b, theta_c be parameter matrices with respective dimensions
  [n**2, k], [n**2, k], [k, n**2]. Then for any matrices a, b with shape [n, n],
  we can form the matrix c with shape [n, n] via the operation:
      ((vec(a) * theta_a) .* (vec(b) * theta_b)) * theta_c = vec(c),  (#)
  where vec(x) is the operator that flattens a matrix with shape [n, n] into a
  row vector with shape [1, n**2], * denotes matrix multiplication and .*
  denotes elementwise multiplication.

  This operation, parameterized by theta_a, theta_b, theta_c, is a matrix
  multiplication algorithm iff c = a*b for all [n, n] matrices a and b. But
  actually it suffices to verify all combinations of one-hot matrices a and b,
  of which there are n**4 such combinations. This gives a batch of n**4 matrix
  triplets (a, b, c) such that equation (#) must hold for each triplet. We solve
  for theta_a, theta_b, theta_c by minimizing the sum of squares of errors
  across this batch.

  Finally, theta_c can be computed from theta_a and theta_b. Therefore it
  suffices to learn theta_a and theta_b, from which theta_c and therefore the
  objective value can be computed.
  """

  def __init__(self, n, k):
    assert isinstance(n, int), "n must be an integer"
    assert isinstance(k, int), "k must be an integer"
    assert n >= 2, "Must have n >= 2"
    assert k >= n**2 and k <= n**3, "Must have n**2 <= k <= n**3"

    param_shapes = [(n**2, k), (n**2, k)]  # theta_a, theta_b
    super(MatMulAlgorithm, self).__init__(
        param_shapes, random_seed=None, noise_stdev=0.0)

    self.n = n
    self.k = k

    # Build a batch of all combinations of one-hot matrices a, b, and their
    # respective products c. Correctness on this batch is a necessary and
    # sufficient condition for the algorithm to be valid. The number of matrices
    # in {a, b, c}_3d is n**4 and each matrix is n x n.
    onehots = np.identity(n**2).reshape(n**2, n, n)
    a_3d = np.repeat(onehots, n**2, axis=0)
    b_3d = np.tile(onehots, [n**2, 1, 1])
    c_3d = np.matmul(a_3d, b_3d)

    # Convert the batch to 2D Tensors.
    self.a = tf.constant(a_3d.reshape(n**4, n**2), tf.float32, name="a")
    self.b = tf.constant(b_3d.reshape(n**4, n**2), tf.float32, name="b")
    self.c = tf.constant(c_3d.reshape(n**4, n**2), tf.float32, name="c")

  def init_tensors(self, seed=None):
    # Initialize params such that the columns of theta_a and theta_b have L2
    # norm 1.
    def _param_initializer(shape, seed=None):
      x = tf.random_normal(shape, dtype=tf.float32, seed=seed)
      return tf.transpose(tf.nn.l2_normalize(tf.transpose(x), 1))

    return [_param_initializer(shape, seed) for shape in self.param_shapes]

  def objective(self, parameters, data=None, labels=None):
    theta_a = parameters[0]
    theta_b = parameters[1]

    # Compute theta_c from theta_a and theta_b.
    p = tf.matmul(self.a, theta_a) * tf.matmul(self.b, theta_b)
    p_trans = tf.transpose(p, name="p_trans")
    p_inv = tf.matmul(
        tf.matrix_inverse(tf.matmul(p_trans, p)), p_trans, name="p_inv")
    theta_c = tf.matmul(p_inv, self.c, name="theta_c")

    # Compute the "predicted" value of c.
    c_hat = tf.matmul(p, theta_c, name="c_hat")

    # Compute the loss (sum of squared errors).
    loss = tf.reduce_sum((c_hat - self.c)**2, name="loss")

    return loss


def matmul_problem_sequence(n, k_min, k_max):
  """Helper to generate a sequence of matrix multiplication problems."""
  return [(_Spec(MatMulAlgorithm, (n, k), {}), None, None)
          for k in range(k_min, k_max + 1)]


def init_fixed_variables(arrays):
  with tf.variable_scope(PARAMETER_SCOPE):
    params = [tf.Variable(arr.astype("float32")) for arr in arrays]
  return params


def _mesh(xlim, ylim, n):
  """Creates a 2D meshgrid covering the given ranges.

  Args:
    xlim: int that defines the desired x-range (-xlim, xlim)
    ylim: int that defines the desired y-range (-ylim, ylim)
    n: number of points in each dimension of the mesh

  Returns:
    xm: 2D array of x-values in the mesh
    ym: 2D array of y-values in the mesh
  """
  return np.meshgrid(np.linspace(-xlim, xlim, n),
                     np.linspace(-ylim, ylim, n))







#########################################################
class PreResNet_QNN(SoftmaxClassifier):
  """Builds an PreResNet with bottleneck block."""
  def __init__(self,
              image_shape,
              n_classes,
              cfg_path,
              select_path,
              depth=20,
              activation=tf.nn.relu,
              random_seed=None,
              noise_stdev=0.0,
              random_sparse_method=None,
              random_sparse_prob=[1.0]):
    print("It is not QNN, it is the backup of full precision ResNet")
    exit()
    expansion = 4
    n_channels, px, py = image_shape
    self.n_channels = n_channels
    # Store the activation.
    self.activation = activation
    self.random_sparse_method = random_sparse_method
    self.random_sparse_prob = random_sparse_prob
    with open(cfg_path, "rb") as cfg_file:
      self.cfg = pickle.load(cfg_file)
    with open(select_path, "rb") as select_file:
      self.select = pickle.load(select_file)
    self.depth = depth

    assert (self.depth - 2) % 9 == 0, 'depth should be 9n+2'
    n = (depth - 2) // 9
    # params shapes, corresponding to ResNet structure
    param_shapes = []
    self.in_planes = n_channels
    # the first Conv params without bias
    param_shapes.append((3,3,self.in_planes,16))
    self.in_planes = 16
    # adding block
    param_shapes.extend(self._make_layer_shape(16, n, cfg = self.cfg[0:3*n]))
    param_shapes.extend(self._make_layer_shape(32, n, cfg = self.cfg[3*n:6*n], stride=2))
    param_shapes.extend(self._make_layer_shape(64, n, cfg = self.cfg[6*n:9*n], stride=2))
    # The BN after the final block
    param_shapes.append((64 * expansion,))
    param_shapes.append((64 * expansion,))
    # adding Linear layer
    param_shapes.append((self.cfg[-1], n_classes)) # for bottleneck block the 512 here should be 512*4
    param_shapes.append((n_classes, ))
    super(PreResNet, self).__init__(param_shapes, random_seed, noise_stdev)

  # Another implementation of gradient that are more friendly to deeper networks
  def gradients(self, objective, parameters):
    """Compute gradients of the objective with respect to the parameters.

    Args:
      objective: The objective op (e.g. output of self.objective())
      parameters: A list of tensors (the parameters to optimize)

    Returns:
      A list of tensors representing the gradient for each parameter,
        returned in the same order as the given list
    """
    if self.random_sparse_method == "layer_wise" or self.random_sparse_method is None:
      _random_sparse_prob = self.random_sparse_prob
    else:
      _random_sparse_prob = [1.0]

    def real_gradient(p):
      return tf.gradients(objective, parameter)[0]
    def fake_gradient(p):
      return tf.constant(0.0, shape=parameter.shape, dtype=tf.float32)

    parameters_list = list(parameters)
    grads = []
    grad_flag_list = []
    revised_grad_flag_list = []
    expand_random_sparse_prob = expand_list(len(parameters_list), self.random_sparse_prob)
    assert len(parameters_list) == len(expand_random_sparse_prob), ("Unsuccessful expand")
    for rd_ratio in expand_random_sparse_prob:
      rd = tf.random.uniform(shape=[], maxval=1)
      grad_flag = tf.math.less_equal(rd, rd_ratio)
      grad_flag_list.append(tf.expand_dims(grad_flag, 0)) # expand dims
    grad_flag_tensor = tf.concat(grad_flag_list, axis=0)
    grad_flag_tensor_indices = tf.sort(tf.squeeze(tf.where(grad_flag_tensor)))
    grad_flag_BP_stop_index = grad_flag_tensor_indices[0]
    for i, parameter in enumerate(parameters_list):
      weight_index_tensor = tf.constant(i,dtype=tf.int64)
      grad_flag = tf.math.greater_equal(weight_index_tensor, grad_flag_BP_stop_index)
      grad_to_append = tf.cond(grad_flag, lambda: real_gradient(parameter), lambda: fake_gradient(parameter))
      revised_grad_flag_list.append(grad_flag)
      grads.append(grad_to_append)

    noisy_grads = []

    for grad in grads:
      if isinstance(grad, tf.IndexedSlices):
        noise = self.noise_stdev * tf.random_normal(tf.shape(grad.values))
        new_grad = tf.IndexedSlices(grad.values + noise, grad.indices)
      else:
        new_grad = grad + self.noise_stdev * tf.random_normal(grad.get_shape())
      noisy_grads.append(new_grad)

    return noisy_grads, revised_grad_flag_list

  def _make_layer_shape(self, planes, num_blocks, cfg, stride=1):
    expansion = 4
    layer_shape = []
    layer_shape.extend(PreResNet_Bottleneck_shape(self.in_planes, planes, cfg[0:3], stride))
    self.in_planes = planes*expansion
    for i in range(1, num_blocks):
      layer_shape.extend(PreResNet_Bottleneck_shape(self.in_planes, planes, cfg[3*i: 3*(i+1)]))
    return layer_shape

  def _make_layer_init(self, planes, num_blocks, stride=1):
    expansion = 4
    layer_init = []
    layer_init.extend(self.bottleneck_init(self.init_in_planes, planes, stride))
    self.init_in_planes = planes*expansion
    for i in range(1, num_blocks):
      layer_init.extend(self.bottleneck_init(self.init_in_planes, planes))
    return layer_init

  def bottleneck_init(self, in_planes, planes, stride=1):
    expansion = 4
    block_init = []
    # BN 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 1
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    # BN 2
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 2
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    # BN 3
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    block_init.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Conv 3
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
    self.init_index += 1
    if stride != 1 or in_planes != planes * expansion:
      # downsample Conv
      shape = self.param_shapes[self.init_index]
      fan_in = float(shape[0]*shape[1]*shape[2])
      block_init.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=self.seed))
      self.init_index += 1
    return block_init
  def init_tensors(self, seed=None, pretrained_model_path=None):
    """Return a list of tensors with the given shape."""
    self.seed = seed
    init_params = []
    self.init_in_planes = self.n_channels
    self.init_index = 0
    # The first Conv
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[0]*shape[1]*shape[2])
    init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
    self.init_index += 1
    self.init_in_planes = 16
    # adding block init
    n = (self.depth - 2) // 9
    init_params.extend(self._make_layer_init(16, n))
    init_params.extend(self._make_layer_init(32, n, stride=2))
    init_params.extend(self._make_layer_init(64, n, stride=2))
    # the BN after final block
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.ones(shape)) # gamma
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.zeros(shape)) # beta
    self.init_index += 1
    # Linear layer
    shape = self.param_shapes[self.init_index]
    fan_in = float(shape[1])
    init_params.append(tf.random_normal(shape, mean=0., stddev=math.sqrt(2.0/fan_in), seed=seed))
    self.init_index += 1
    shape = self.param_shapes[self.init_index]
    init_params.append(tf.zeros(shape))
    if pretrained_model_path is not None:
      with open(pretrained_model_path, "rb") as params_file:
        pretrained_params = pickle.load(params_file)
      for k_id, k in enumerate(pretrained_params):
        # only load before the last FC layer
        if (k.shape[-1] != 5) or (k_id < len(pretrained_params) - 2):
          init_params[k_id] = k
          print("Loading weight shape:", k.shape)
        else:
          print("Not loading weight shape:", k.shape)
    return init_params

  def _make_layer_imp(self, out, params, planes, num_blocks, cfg, stride=1):
    expansion = 4
    out = self.bottleneck_imp(out, params, self.imp_in_planes, planes, cfg[0:3], stride)
    self.imp_in_planes = planes*expansion
    for i in range(1, num_blocks):
      out = self.bottleneck_imp(out, params, self.imp_in_planes, planes, cfg[3*i: 3*(i+1)])
    return out

  def bottleneck_imp(self, _input, params, in_planes, planes, cfg, stride=1):
    expansion = 4
    # BN 1
    out = _input
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    # select
    selected = self.select[self.select_index]
    self.select_index += 1
    selected = np.squeeze(np.argwhere(selected))
    if selected.size == 1:
      selected = np.resize(selected, (1,))
    out = tf.gather(out,selected,axis=3)
    # relu
    out = self.activation(out)
    # Conv 1
    w_conv = params[self.imp_index]
    self.imp_index += 1
    out = tf.nn.conv2d(out, w_conv, strides=[1]*4, padding="SAME")
    # BN 2
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    # relu
    out = self.activation(out)
    # Conv 2
    w_conv = params[self.imp_index]
    self.imp_index += 1
    out = tf.nn.conv2d(out, w_conv, strides=[1, stride, stride, 1], padding="SAME")
    # BN 3
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    # relu
    out = self.activation(out)
    # Conv 3
    w_conv = params[self.imp_index]
    self.imp_index += 1
    out = tf.nn.conv2d(out, w_conv, strides=[1]*4, padding="SAME")
    # downsample
    if stride != 1 or in_planes != planes * expansion:
      # downsample Conv
      w_conv = params[self.imp_index]
      self.imp_index += 1
      shortcut = tf.nn.conv2d(_input, w_conv, strides=[1, stride, stride, 1], padding="SAME")
    else:
      shortcut = _input
    out = out + shortcut

    return out

  def inference(self, params, data, is_training=tf.squeeze(tf.constant([True], dtype=tf.bool))):
    self.is_training = is_training
    conv_input = data
    self.imp_in_planes = self.n_channels
    self.imp_index = 0
    self.select_index = 0 # the index to indicate the current select
    # imp: first Conv
    w_conv = params[self.imp_index]
    self.imp_index += 1
    out = tf.nn.conv2d(conv_input, w_conv, strides=[1]*4, padding="SAME")
    self.imp_in_planes = 16
    # add block implementation
    n = (self.depth - 2) // 9
    out = self._make_layer_imp(out, params, 16, n , cfg = self.cfg[0:3*n])
    out = self._make_layer_imp(out, params, 32, n , cfg = self.cfg[3*n:6*n], stride=2)
    out = self._make_layer_imp(out, params, 64, n , cfg = self.cfg[6*n:9*n], stride=2)
    # add BN after the final block
    gamma_bn = params[self.imp_index]
    self.imp_index += 1
    beta_bn = params[self.imp_index]
    self.imp_index += 1
    out = tf.cond(self.is_training, lambda: batch_norm_training(gamma_bn, beta_bn, out), lambda: batch_norm_inference(gamma_bn, beta_bn, out))
    # select here
    selected = self.select[self.select_index]
    self.select_index += 1
    selected = np.squeeze(np.argwhere(selected))
    if selected.size == 1:
      selected = np.resize(selected, (1,))
    out = tf.gather(out,selected,axis=3)
    # relu
    out = self.activation(out)
    # avg_pool
    out = tf.reduce_mean(out, axis=[1,2])
    # FC
    w_fc = params[self.imp_index]
    self.imp_index += 1
    bias_fc = params[self.imp_index]
    self.imp_index += 1
    # flatten
    # print("out before flatten: ", out.shape)
    out = tf.reshape(out, [-1, w_fc.shape[0]])
    out = tf.nn.bias_add(tf.matmul(out, w_fc), bias_fc)
    return out