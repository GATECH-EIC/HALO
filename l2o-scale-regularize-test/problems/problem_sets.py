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

"""Groups of problems of different types for optimizer training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from problems import datasets
from problems import model_adapter
from problems import problem_generator as pg
from problems import problem_spec

_Spec = problem_spec.Spec

mnist_config = {
    'A': [16, 'M', 32, 'M'],
}

def quadratic_problems():
  return [
      (_Spec(pg.Quadratic, (20,), {}), None, None),
      (_Spec(pg.Quadratic, (25,), {}), None, None),
      (_Spec(pg.Quadratic, (50,), {}), None, None),
      (_Spec(pg.Quadratic, (100,), {}), None, None),
  ]

def mnist_conv_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
             {"activation": tf.nn.relu}), datasets.mnist(train=True), 64)
  ]

# def mnist_conv_problems():
#   return [
#       # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
#       #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
#       (_Spec(pg.ConvNet_pooling, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)], [1, 2]),
#              {"activation": tf.nn.relu}), datasets.mnist(train=True), 64)
#   ]
def cifar10_conv_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 10, [(3, 3, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu}), 'cifar10', 64)
  ]

def test_mnist_conv_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
             {"activation": tf.nn.relu}), 'mnist', 64)
  ]

def pretrain_mnist_conv_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((1, 28, 28), 5, [(3, 3, 8), (5, 5, 8)]),
             {"activation": tf.nn.relu}), 'mnist-B', 64)
  ]

def pretrain_TDD_MLP_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.MLP, (26, 4, [30,30]),
             {"activation": tf.nn.relu}), 'TDD-Male', 64)
  ]


def pretrain_PIE_conv_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((1, 64, 64), 34, [(3, 3, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu}), 'PIE-B', 64)
  ]

def pretrain_mnist_conv_problems_wide():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((1, 28, 28), 5, [(3, 3, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu}), 'mnist-B', 64)
  ]

def test_cifar10_conv_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 10, [(3, 3, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu}), 'cifar10', 64)
  ]

def pretrain_cifar10_conv_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_sin_lstm_problems():
    return [
        # (_Spec(pg.SinLSTMModel, (128, 20, 10, 1), {}), 'lstm-sin-G', 128)
        (_Spec(pg.SinLSTMModel, (128, 256, 128, 2), {}), 'lstm_UCI_HAR_Dataset', 128)
    ]

# def pretrain_conv_lstm_problems():
#   return [
#       (_Spec(pg.Conv_LSTM, (128, 100, 128, 1, [(3, 128, 64), (3, 64, 64)]), {}), 'lstm_UCI_HAR_Dataset-B', 128)
#   ]

####################################################

def pretrain_conv_lstm_problems():
  return [
      (_Spec(pg.Conv_LSTM, (64, 100, 128, 1, [(3, 128, 64), (3, 64, 64), (1, 64, 64)]), {}), 'lstm_UCI_HAR_Dataset-B', 64)
  ]

def pretrain_conv_lstm_5_layer_problems():
  return [
      (_Spec(pg.Conv_LSTM, (64, 100, 128, 1, [(3, 128, 64), (1, 64, 64), (1, 64, 64), (3, 64, 64), (1, 64, 64)]), {}), 'lstm_UCI_HAR_Dataset-B', 64)
  ]

def pretrain_conv_UCI_problems():
  return [
      (_Spec(pg.Conv_UCI, (64, [(3, 128, 64), (3, 64, 64), (1, 64, 64)]), {}), 'lstm_UCI_HAR_Dataset-B', 64)
  ]

# def pretrain_conv_lstm_WISDM_problems():
#   return [
#       (_Spec(pg.Conv_LSTM, (64, 100, 128, 1, [(3, 180, 64), (1, 64, 64), (1, 64, 64)], 6), {}), 'lstm_WISDM_HAR_Dataset-B', 64)
#   ]

def pretrain_conv_lstm_WISDM_problems():
  return [
      (_Spec(pg.Conv_LSTM_v2, (64, 30, [], 6), {}), 'lstm_WISDM_HAR_Dataset-B', 64)
  ]

def pretrain_lstm_SML_problems():
  return [
      (_Spec(pg.Conv_LSTM_v3, (64, 10, 128, 1, [(1, 3, 64), (1, 64, 64), (1, 64, 64)]), {}), 'lstm_SML_Dataset-B', 64)
  ]

def pretrain_MLP_GAS_problems():
  return [
      (_Spec(pg.MLP, (129, 3, [30, 30]),
             {"activation": tf.nn.relu}),
       'MLP_GAS_Dataset-B', 64)
  ]
####################################################

def pretrain_cifar10_resnet18():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ResNet18, ((3, 32, 32), 5, [2,2,2,2]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]


def pretrain_cifar10_conv_problems_v2():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 8), (3, 3, 8), (5, 5, 8), (5, 5, 8)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_cifar10_conv_problems_v3():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 16), (3, 3, 16), (5, 5, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_cifar10_conv_problems_v4():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 32), (3, 3, 32), (5, 5, 32), (5, 5, 32)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_cifar10_conv_problems_v5():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 32), (3, 3, 32), (3, 3, 32), (5, 5, 32), (5, 5, 32), (5, 5, 32)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_cifar10_conv_problems_v6():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 16), (3, 3, 16), (3, 3, 16), (3, 3, 16), (5, 5, 16), (5, 5, 16), (5, 5, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_cifar10_conv_problems_v7():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 32), (3, 3, 32), (3, 3, 32), (3, 3, 32), (5, 5, 32), (5, 5, 32), (5, 5, 32), (5, 5, 32)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_cifar10_conv_problems_v8():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 64), (3, 3, 64), (3, 3, 64), (5, 5, 64), (5, 5, 64), (5, 5, 64)]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def adapt_sin_lstm_problems():
    return [
        (_Spec(pg.SinLSTMModel, (128, 20, 10, 1), {}), 'lstm-sin-U', 128)
    ]

# def adapt_conv_lstm_problems(batch_size=128, random_sparse_method=None, random_sparse_prob=[1.0]):
#   return [
#       (_Spec(pg.Conv_LSTM, (128, 100, 128, 1, [(3, 128, 64), (3, 64, 64)]), {"random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'lstm_UCI_HAR_Dataset-A', batch_size)
#   ]

####################################################

def adapt_conv_lstm_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      (_Spec(pg.Conv_LSTM, (64, 100, 128, 1, [(3, 128, 64), (3, 64, 64), (1, 64, 64)]), {"random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'lstm_UCI_HAR_Dataset-A', batch_size)
  ]

def adapt_conv_lstm_5_layer_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      (_Spec(pg.Conv_LSTM, (64, 100, 128, 1, [(3, 128, 64), (1, 64, 64), (1, 64, 64), (3, 64, 64), (1, 64, 64)]), {"random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'lstm_UCI_HAR_Dataset-A', batch_size)
  ]

def adapt_conv_UCI_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      (_Spec(pg.Conv_UCI, (64, [(3, 128, 64), (1, 64, 64), (3, 64, 64)]), {"random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'lstm_UCI_HAR_Dataset-A', batch_size)
  ]

def adapt_conv_lstm_SML(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      (_Spec(pg.Conv_LSTM_v3, (64, 10, 64, 1, [(1, 3, 64), (1, 64, 64), (1, 64, 64)]), {"random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'lstm_SML_Dataset-A', batch_size)
  ]

# def adapt_MLP_GAS_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
#   return [
#       (_Spec(pg.Conv_LSTM_v3, (64, 10, 64, 1, [(1, 3, 64), (1, 64, 64), (1, 64, 64)]), {"random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'lstm_SML_Dataset-A', batch_size)
#   ]

def adapt_MLP_GAS_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      (_Spec(pg.MLP, (129, 3, [30, 30]),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}),
       'MLP_GAS_Dataset-A', 64)
  ]
####################################################

def adapt_PIE_conv_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((1, 64, 64), 34, [(3, 3, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'PIE-A', batch_size)
  ]


def adapt_cifar10_conv_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'cifar10-A', batch_size)
  ]

def adapt_TDD_MLP_problems(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.MLP, (26, 4, [30,30]),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'TDD-Female', batch_size)
  ]

def adapt_cifar10_conv_problems_v2(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 8), (3, 3, 8), (5, 5, 8), (5, 5, 8)]),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'cifar10-A', batch_size)
  ]

def adapt_cifar10_conv_problems_v3(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ConvNet, ((3, 32, 32), 5, [(3, 3, 16), (3, 3, 16), (5, 5, 16), (5, 5, 16)]),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'cifar10-A', batch_size)
  ]

def adapt_cifar10_resnet18(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.ResNet18, ((3, 32, 32), 5, [2,2,2,2]),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'cifar10-A', batch_size)
  ]

def adapt_cifar10_preresnet20_p70(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.PreResNet, ((3, 32, 32), 5,
      "/chaojian_shared_datasets/rethinking-network-pruning/cifar/network-slimming/prune_0.7_cfg.pickle",
      "/chaojian_shared_datasets/rethinking-network-pruning/cifar/network-slimming/prune_0.7_select.pickle"),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'cifar10-A', batch_size)
  ]

def adapt_cifar10_preresnet20_p80(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.PreResNet, ((3, 32, 32), 5,
      "/chaojian_shared_datasets/rethinking-network-pruning/cifar/network-slimming/prune_0.8_cfg.pickle",
      "/chaojian_shared_datasets/rethinking-network-pruning/cifar/network-slimming/prune_0.8_select.pickle"),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'cifar10-A', batch_size)
  ]

def adapt_cifar10_preresnet20_p90(batch_size=64, random_sparse_method=None, random_sparse_prob=[1.0]):
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.PreResNet, ((3, 32, 32), 5,
      "/chaojian_shared_datasets/rethinking-network-pruning/cifar/network-slimming/prune_0.9_cfg.pickle",
      "/chaojian_shared_datasets/rethinking-network-pruning/cifar/network-slimming/prune_0.9_select.pickle"),
             {"activation": tf.nn.relu, "random_sparse_method": random_sparse_method, "random_sparse_prob": random_sparse_prob}), 'cifar10-A', batch_size)
  ]
# def cifar10_conv_problems():
#   return [
#       # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
#       #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
#       (_Spec(pg.ConvNet_pooling, ((3, 32, 32), 10, [(3, 3, 32), (3, 3, 32)], [1, 2]),
#              {"activation": tf.nn.relu}), 'cifar10', 64)
#   ]

def mnist_mlp_problems():
  tp = [
      (_Spec(pg.FullyConnected, (784, 10),
             {"hidden_sizes": (20,), "activation": tf.nn.relu}),
       datasets.mnist(train=True), 64)
  ]
  return tp

def vgg16_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.vgg16, ((3, 32, 32), 10, cfgs['D'], [2, 4, 7, 10, 13]),
             {"activation": tf.nn.relu}), 'cifar10', 64)
  ]


def adapt_vgg16_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.vgg16, ((3, 32, 32), 5, cfgs['D'], [2, 4, 7, 10, 13]),
             {"activation": tf.nn.relu}), 'cifar10-A', 64)
  ]

def pretrain_vgg16_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.vgg16, ((3, 32, 32), 5, cfgs['D'], [2, 4, 7, 10, 13]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

def pretrain_vgg11_problems():
  return [
      # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
      #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
      (_Spec(pg.vgg16, ((3, 32, 32), 5, cfgs['A'], [1, 2, 4, 6, 8]),
             {"activation": tf.nn.relu}), 'cifar10-B', 64)
  ]

cfgs = {
    'A': [64, 128, 256, 256, 512, 512, 512, 512],
    'B': [64, 64, 128, 128, 256, 256, 512, 512, 512, 512],
    'D': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'E': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}

# def vgg_problems():
#   return [
#       # (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
#       #        {"activation": tf.nn.sigmoid}), datasets.mnist(train=True), 64),
#       (_Spec(pg.ConvNet, ((1, 32, 32), 10, cfgs['D']),
#              {"activation": tf.nn.relu}), datasets.cifar(train=False), 64)
#   ]
#
#
# cfgs = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
# def ConvNet_problems():
#   return [
#       (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
#              {"activation": tf.nn.sigmoid}), datasets.mnist(), 64,
#        "convnet_mnist_sigmoid", None),
#       (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
#              {"activation": tf.nn.relu}), datasets.mnist(), 64,
#        "convnet_mnist_relu", None),
#   ]
# Note: this group contains one non-noisy problem for historical reasons. The
# original training set before the refactor included this set of quadratics.
def quadratic_problems_noisy():
  return [
      (_Spec(pg.Quadratic, (20,), {"noise_stdev": 0.5}), None, None),
      (_Spec(pg.Quadratic, (25,), {"noise_stdev": 0.0}), None, None),
      (_Spec(pg.Quadratic, (50,), {"noise_stdev": 1.0}), None, None),
      (_Spec(pg.Quadratic, (100,), {"noise_stdev": 2.0}), None, None),
  ]


def quadratic_problems_large():
  return [
      (_Spec(pg.Quadratic, (784,), {}), None, None),
      (_Spec(pg.Quadratic, (1024,), {}), None, None),
      (_Spec(pg.Quadratic, (2048,), {}), None, None),
  ]


def bowl_problems():
  return [
      (_Spec(pg.Bowl, (0.1,), {"noise_stdev": 0.0}), None, None),
      (_Spec(pg.Bowl, (1.0,), {"noise_stdev": 0.0}), None, None),
      (_Spec(pg.Bowl, (5.0,), {"noise_stdev": 0.0}), None, None),
      (_Spec(pg.Bowl, (5.0,), {"noise_stdev": 0.0, "angle": np.pi / 4.}),
       None, None),
  ]


def bowl_problems_noisy():
  return [
      (_Spec(pg.Bowl, (0.1,), {"noise_stdev": 0.1}), None, None),
      (_Spec(pg.Bowl, (1.0,), {"noise_stdev": 0.1}), None, None),
      (_Spec(pg.Bowl, (5.0,), {"noise_stdev": 0.1}), None, None),
      (_Spec(pg.Bowl, (5.0,), {"noise_stdev": 0.1, "angle": np.pi / 4.}),
       None, None),
  ]


def sparse_softmax_2_class_sparse_problems():
  return [(_Spec(pg.SparseSoftmaxRegression, (5, 2), {"noise_stdev": 0.0}),
           datasets.noisy_parity_class(5, random_seed=123), 23),]


def one_hot_sparse_softmax_2_class_sparse_problems():
  return [
      (_Spec(pg.OneHotSparseSoftmaxRegression, (5, 2), {"noise_stdev": 0.0}),
       datasets.noisy_parity_class(5, random_seed=123), 23),
  ]


def softmax_2_class_problems():
  return [
      (_Spec(pg.SoftmaxRegression, (10, 2), {}), datasets.random(
          10, 1000, random_seed=123, sep=2.0), 100),
      (_Spec(pg.SoftmaxRegression, (100, 2), {}), datasets.random(
          100, 1000, random_seed=123), 50),
      (_Spec(pg.SoftmaxRegression, (200, 2), {}), datasets.random(
          200, 1000, random_seed=123, sep=1.5), 20),
      (_Spec(pg.SoftmaxRegression, (256, 2), {}), datasets.random(
          256, 1000, random_seed=123, sep=1.5), 100),
  ]


def softmax_2_class_problems_noisy():
  return [
      (_Spec(pg.SoftmaxRegression, (10, 2), {"noise_stdev": 0.5}),
       datasets.random(10, 1000, random_seed=123, sep=2.0), 100),
      (_Spec(pg.SoftmaxRegression, (100, 2), {"noise_stdev": 0.1}),
       datasets.random(100, 1000, random_seed=123), 50),
      (_Spec(pg.SoftmaxRegression, (200, 2), {"noise_stdev": 0.1}),
       datasets.random(200, 1000, random_seed=123, sep=1.5), 20),
      (_Spec(pg.SoftmaxRegression, (256, 2), {"noise_stdev": 0.5}),
       datasets.random(256, 1000, random_seed=123, sep=1.5), 100),
  ]


def optimization_test_problems():
  return [
      (_Spec(pg.Ackley, (), {}), None, None),
      (_Spec(pg.Beale, (), {}), None, None),
      (_Spec(pg.Booth, (), {}), None, None),
      (_Spec(pg.Branin, (), {}), None, None),
      (_Spec(pg.LogSumExp, (), {}), None, None),
      (_Spec(pg.Matyas, (), {}), None, None),
      (_Spec(pg.Michalewicz, (), {}), None, None),
      (_Spec(pg.Rosenbrock, (), {}), None, None),
      (_Spec(pg.StyblinskiTang, (), {}), None, None),
  ]


def optimization_test_problems_noisy():
  return [
      (_Spec(pg.Ackley, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.Beale, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.Booth, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.Branin, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.LogSumExp, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.Matyas, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.Michalewicz, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.Rosenbrock, (), {"noise_stdev": 1.}), None, None),
      (_Spec(pg.StyblinskiTang, (), {"noise_stdev": 1.}), None, None),
  ]


def fully_connected_random_2_class_problems():
  return [
      (_Spec(pg.FullyConnected, (8, 2),
             {"hidden_sizes": (8, 5,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(8, 1000), 10),
      (_Spec(pg.FullyConnected, (12, 2),
             {"hidden_sizes": (8, 5, 3), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(12, 1000), 200),
      (_Spec(pg.FullyConnected, (5, 2),
             {"hidden_sizes": (4, 4, 4, 4,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(5, 1000), 100),
      (_Spec(pg.FullyConnected, (11, 2),
             {"hidden_sizes": (4, 5, 6,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(11, 1000), 64),
      (_Spec(pg.FullyConnected, (9, 2),
             {"hidden_sizes": (8,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(9, 1000), 128),
      (_Spec(pg.FullyConnected, (7, 2),
             {"hidden_sizes": (8, 5,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(7, 1000), 16),
      (_Spec(pg.FullyConnected, (8, 2),
             {"hidden_sizes": (32, 64,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(8, 1000), 10),
      (_Spec(pg.FullyConnected, (12, 2),
             {"hidden_sizes": (16, 8, 3), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(12, 1000), 200),
      (_Spec(pg.FullyConnected, (5, 2),
             {"hidden_sizes": (8, 8, 8, 8,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(5, 1000), 100),
      (_Spec(pg.FullyConnected, (11, 2),
             {"hidden_sizes": (10, 12, 12,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(11, 1000), 64),
      (_Spec(pg.FullyConnected, (9, 2),
             {"hidden_sizes": (32,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(9, 1000), 128),
      (_Spec(pg.FullyConnected, (7, 2),
             {"hidden_sizes": (32, 64,), "activation": tf.nn.sigmoid}),
       datasets.random_mlp(7, 1000), 16),
  ]


def matmul_problems():
  return sum([
      pg.matmul_problem_sequence(2, 5, 8),
      pg.matmul_problem_sequence(3, 19, 24)], [])


def log_objective_problems():
  return [
      (_Spec(pg.LogObjective, [_Spec(pg.Quadratic, (20,), {})], {}),
       None, None),
      (_Spec(pg.LogObjective, [_Spec(pg.Quadratic, (50,), {})], {}),
       None, None),
      (_Spec(pg.LogObjective, [_Spec(pg.Quadratic, (100,), {})], {}),
       None, None),
      (_Spec(pg.LogObjective, [_Spec(pg.Bowl, (0.1,), {})], {}), None, None),
      (_Spec(pg.LogObjective, [_Spec(pg.Bowl, (1.0,), {})], {}), None, None),
      (_Spec(pg.LogObjective, [_Spec(pg.Bowl, (5.0,), {})], {}), None, None),
  ]


def sparse_gradient_problems():
  return [
      (_Spec(pg.SparseProblem, [_Spec(pg.Quadratic, (20,), {})], {}),
       None, None),
      (_Spec(pg.SparseProblem, [_Spec(pg.Quadratic, (50,), {})], {}),
       None, None),
      (_Spec(pg.SparseProblem, [_Spec(pg.Quadratic, (100,), {})], {}),
       None, None),
      (_Spec(pg.SparseProblem, [_Spec(pg.Bowl, (0.1,), {})], {}), None, None),
      (_Spec(pg.SparseProblem, [_Spec(pg.Bowl, (1.0,), {})], {}), None, None),
      (_Spec(pg.SparseProblem, [_Spec(pg.Bowl, (5.0,), {})], {}), None, None),
  ]


def sparse_gradient_problems_mlp():
  return [
      (_Spec(pg.SparseProblem, [
          _Spec(pg.FullyConnected, (8, 2), {
              "hidden_sizes": (8, 5,),
              "activation": tf.nn.sigmoid
          })
      ], {}), datasets.random_mlp(8, 1000), 10),
      (_Spec(pg.SparseProblem, [
          _Spec(pg.FullyConnected, (12, 2), {
              "hidden_sizes": (8, 5, 3),
              "activation": tf.nn.sigmoid
          })
      ], {}), datasets.random_mlp(12, 1000), 200),
      (_Spec(pg.SparseProblem, [
          _Spec(pg.FullyConnected, (5, 2), {
              "hidden_sizes": (4, 4, 4, 4,),
              "activation": tf.nn.sigmoid
          })
      ], {}), datasets.random_mlp(5, 1000), 100),
  ]


def rescale_problems():
  return [
      (_Spec(pg.Rescale, [_Spec(pg.Norm, (18,), {"norm_power": 2.5})],
             {"scale": 0.123}), None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Norm, (18,), {"norm_power": 1.5})],
             {"scale": 8}), None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Norm, (18,), {"norm_power": 2.})],
             {"scale": 50}), None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Norm, (18,), {"norm_power": 3.})],
             {"scale": 200}), None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Norm, (18,), {"norm_power": 1.})],
             {"scale": 1000}), None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Quadratic, (20,), {})], {"scale": 0.1}),
       None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Quadratic, (25,), {})], {"scale": 10.}),
       None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Quadratic, (50,), {})], {"scale": 350.}),
       None, None),
      (_Spec(pg.Rescale, [_Spec(pg.Quadratic, (100,), {})], {"scale": 132}),
       None, None),
  ]


def norm_problems():
  return [
      # < 1 Norm causes NaN gradients early in training.
      (_Spec(pg.Norm, (27,), {"norm_power": 1.}), None, None),
      (_Spec(pg.Norm, (25,), {"norm_power": 2.}), None, None),
      (_Spec(pg.Norm, (22,), {"norm_power": 3.}), None, None),
  ]


def norm_problems_noisy():
  return [
      # < 1 Norm causes NaN gradients early in training.
      (_Spec(pg.Norm, (19,), {"noise_stdev": .1, "norm_power": 1.}),
       None, None),
      (_Spec(pg.Norm, (26,), {"noise_stdev": .1, "norm_power": 2.}),
       None, None),
      (_Spec(pg.Norm, (23,), {"noise_stdev": .1, "norm_power": 3.}),
       None, None),
  ]


def sum_problems():
  return [
      (_Spec(pg.SumTask, [[
          _Spec(pg.Quadratic, (11,), {}),
          _Spec(pg.Quadratic, (3,), {}),
          _Spec(pg.Quadratic, (9,), {}),
          _Spec(pg.Quadratic, (7,), {}),
          _Spec(pg.Quadratic, (5,), {}),
          _Spec(pg.Quadratic, (13,), {}),
          _Spec(pg.Quadratic, (12,), {})
      ]], {}), None, None),
      (_Spec(pg.SumTask, [[
          _Spec(pg.Norm, (18,), {"norm_power": 3}),
          _Spec(pg.Quadratic, (25,), {}),
          _Spec(pg.Rosenbrock, (), {})
      ]], {}), None, None),
      (_Spec(pg.SumTask, [[
          _Spec(pg.Rosenbrock, (), {}),
          _Spec(pg.LogSumExp, (), {}),
          _Spec(pg.Ackley, (), {}),
          _Spec(pg.Beale, (), {}),
          _Spec(pg.Booth, (), {}),
          _Spec(pg.StyblinskiTang, (), {}),
          _Spec(pg.Matyas, (), {}),
          _Spec(pg.Branin, (), {}),
          _Spec(pg.Michalewicz, (), {})
      ]], {}), None, None),
      (_Spec(pg.SumTask, [[
          _Spec(pg.Rosenbrock, (), {}),
          _Spec(pg.LogSumExp, (), {}),
          _Spec(pg.Ackley, (), {}),
          _Spec(pg.Beale, (), {}),
          _Spec(pg.Booth, (), {}),
          _Spec(pg.StyblinskiTang, (), {}),
          _Spec(pg.Matyas, (), {}),
          _Spec(pg.Branin, (), {}),
          _Spec(pg.Michalewicz, (), {}),
          _Spec(pg.Quadratic, (5,), {}),
          _Spec(pg.Quadratic, (13,), {})
      ]], {}), None, None),
      (_Spec(pg.SumTask, [[
          _Spec(pg.Quadratic, (11,), {}),
          _Spec(pg.Quadratic, (3,), {})
      ]], {}), None, None),
      (_Spec(pg.SumTask, [[
          _Spec(pg.Rosenbrock, (), {}),
          _Spec(pg.LogSumExp, (), {}),
          _Spec(pg.Ackley, (), {})
      ]], {}), None, None),
  ]


def sum_problems_noisy():
  return [
      (_Spec(pg.SumTask, [[
          _Spec(pg.Quadratic, (11,), {"noise_stdev": 0.1}),
          _Spec(pg.Quadratic, (3,), {"noise_stdev": 0.1}),
          _Spec(pg.Quadratic, (9,), {"noise_stdev": 0.1}),
          _Spec(pg.Quadratic, (7,), {"noise_stdev": 0.1}),
          _Spec(pg.Quadratic, (5,), {"noise_stdev": 0.1}),
          _Spec(pg.Quadratic, (13,), {"noise_stdev": 0.1}),
          _Spec(pg.Quadratic, (12,), {"noise_stdev": 0.1})
      ]], {}), None, None),
      (_Spec(pg.SumTask, [[
          _Spec(pg.Rosenbrock, (), {}),
          _Spec(pg.LogSumExp, (), {}),
          _Spec(pg.Ackley, (), {}),
          _Spec(pg.Beale, (), {}),
          _Spec(pg.Booth, (), {}),
          _Spec(pg.StyblinskiTang, (), {}),
          _Spec(pg.Matyas, (), {}),
          _Spec(pg.Branin, (), {}),
          _Spec(pg.Michalewicz, (), {}),
          _Spec(pg.Quadratic, (5,), {}),
          _Spec(pg.Quadratic, (13,), {"noise_stdev": 0.5})
      ]], {}), None, None),
  ]


def dependency_chain_problems():
  return [
      (_Spec(pg.DependencyChain, (20,), {}), datasets.random_binary(
          20, 1000), 100),
      (_Spec(pg.DependencyChain, (12,), {}), datasets.random_binary(
          12, 200), 10),
      (_Spec(pg.DependencyChain, (56,), {}), datasets.random_binary(
          56, 5000), 100),
      (_Spec(pg.DependencyChain, (64,), {}), datasets.random_binary(
          64, 1000), 50),
      (_Spec(pg.DependencyChain, (13,), {}), datasets.random_binary(
          13, 10000), 50),
      (_Spec(pg.DependencyChain, (20,), {}), datasets.random_binary(
          20, 1000), 128),
      (_Spec(pg.DependencyChain, (12,), {}), datasets.random_binary(
          12, 300), 16),
      (_Spec(pg.DependencyChain, (56,), {}), datasets.random_binary(
          56, 5000), 128),
      (_Spec(pg.DependencyChain, (64,), {}), datasets.random_binary(
          64, 1000), 64),
      (_Spec(pg.DependencyChain, (13,), {}), datasets.random_binary(
          13, 10000), 32),
  ]


def outward_snake_problems():
  return [
      (_Spec(pg.OutwardSnake, (20,), {}), datasets.random_binary(
          20, 1000), 100),
      (_Spec(pg.OutwardSnake, (12,), {}), datasets.random_binary(
          12, 200), 10),
      (_Spec(pg.OutwardSnake, (56,), {}), datasets.random_binary(
          56, 5000), 100),
      (_Spec(pg.OutwardSnake, (64,), {}), datasets.random_binary(
          64, 1000), 50),
      (_Spec(pg.OutwardSnake, (13,), {}), datasets.random_binary(
          13, 10000), 50),
      (_Spec(pg.OutwardSnake, (20,), {}), datasets.random_binary(
          20, 1000), 128),
      (_Spec(pg.OutwardSnake, (12,), {}), datasets.random_binary(
          12, 300), 16),
      (_Spec(pg.OutwardSnake, (56,), {}), datasets.random_binary(
          56, 5000), 128),
      (_Spec(pg.OutwardSnake, (64,), {}), datasets.random_binary(
          64, 1000), 64),
      (_Spec(pg.OutwardSnake, (13,), {}), datasets.random_binary(
          13, 10000), 32),
  ]


def min_max_well_problems():
  return [
      (_Spec(pg.MinMaxWell, (20,), {}), None, None),
      (_Spec(pg.MinMaxWell, (12,), {}), None, None),
      (_Spec(pg.MinMaxWell, (56,), {}), None, None),
      (_Spec(pg.MinMaxWell, (64,), {}), None, None),
      (_Spec(pg.MinMaxWell, (13,), {}), None, None),
  ]


def sum_of_quadratics_problems():
  return [
      (_Spec(pg.SumOfQuadratics, (20,), {}),
       datasets.random_symmetric(20, 1000), 100),
      (_Spec(pg.SumOfQuadratics, (12,), {}),
       datasets.random_symmetric(12, 100), 10),
      (_Spec(pg.SumOfQuadratics, (56,), {}),
       datasets.random_symmetric(56, 5000), 100),
      (_Spec(pg.SumOfQuadratics, (64,), {}),
       datasets.random_symmetric(64, 1000), 50),
      (_Spec(pg.SumOfQuadratics, (13,), {}),
       datasets.random_symmetric(13, 10000), 50),
      (_Spec(pg.SumOfQuadratics, (20,), {}),
       datasets.random_symmetric(20, 1000), 128),
      (_Spec(pg.SumOfQuadratics, (12,), {}),
       datasets.random_symmetric(12, 100), 16),
      (_Spec(pg.SumOfQuadratics, (56,), {}),
       datasets.random_symmetric(56, 5000), 128),
      (_Spec(pg.SumOfQuadratics, (64,), {}),
       datasets.random_symmetric(64, 1000), 64),
      (_Spec(pg.SumOfQuadratics, (13,), {}),
       datasets.random_symmetric(13, 10000), 32),
  ]


def projection_quadratic_problems():
  return [
      (_Spec(pg.ProjectionQuadratic, (20,), {}),
       datasets.random_symmetric(20, 1000), 100),
      (_Spec(pg.ProjectionQuadratic, (12,), {}),
       datasets.random_symmetric(12, 100), 10),
      (_Spec(pg.ProjectionQuadratic, (56,), {}),
       datasets.random_symmetric(56, 5000), 100),
      (_Spec(pg.ProjectionQuadratic, (64,), {}),
       datasets.random_symmetric(64, 1000), 50),
      (_Spec(pg.ProjectionQuadratic, (13,), {}),
       datasets.random_symmetric(13, 10000), 50),
      (_Spec(pg.ProjectionQuadratic, (20,), {}),
       datasets.random_symmetric(20, 1000), 128),
      (_Spec(pg.ProjectionQuadratic, (12,), {}),
       datasets.random_symmetric(12, 100), 16),
      (_Spec(pg.ProjectionQuadratic, (56,), {}),
       datasets.random_symmetric(56, 5000), 128),
      (_Spec(pg.ProjectionQuadratic, (64,), {}),
       datasets.random_symmetric(64, 1000), 64),
      (_Spec(pg.ProjectionQuadratic, (13,), {}),
       datasets.random_symmetric(13, 10000), 32),
  ]


def adapter_rosenbrock_local():
  return [(_Spec(model_adapter.ModelAdapter,
                 (pg.make_rosenbrock_loss_and_init,), {}), None, None),]


def adapter_rosenbrock_worker():
  return [(_Spec(model_adapter.ModelAdapter,
                 (pg.make_rosenbrock_loss_and_init,),
                 {"device": "/job:worker"}), None, None),]


def _test_problem_mlp_scaled_init_small():
  return [
      np.random.randn(10, 32) * np.sqrt(2./10),
      np.random.randn(32,) * 0.1,
      np.random.randn(32, 64) * np.sqrt(2./32.),
      np.random.randn(64,) * 0.1,
      np.random.randn(64, 2) * np.sqrt(2./64.),
      np.random.randn(2,) * 0.1
  ]


def _test_problem_mlp_scaled_init_large():
  return [
      np.random.randn(20, 32) * np.sqrt(2./20),
      np.random.randn(32,) * 0.1,
      np.random.randn(32, 64) * np.sqrt(2./32.),
      np.random.randn(64,) * 0.1,
      np.random.randn(64, 10) * np.sqrt(2./64.),
      np.random.randn(10,) * 0.1
  ]


def _test_problem_mlp_scaled_init_mnist():
  return [
      np.random.randn(784, 64) * np.sqrt(2./784.),
      np.random.randn(64,) * 0.1,
      np.random.randn(64, 10) * np.sqrt(2./ 64.),
      np.random.randn(10,) * 0.1,
  ]


# Wrap this construction in a function to avoid UnparsedFlagAccessError
def test_problems():
  """Test problems for visualizations."""
  # Unlike the training problem sets, these test problems are made up of
  # length-5 tuples. The final items in the tuple are the name of the problem
  # and the initialization random_seed for testing consistency.
  tp = [
      (_Spec(pg.Quadratic, (20,), {"random_seed": 1234}), None, None,
       "quad_problem", 5678),
      (_Spec(pg.Quadratic, (20,), {"noise_stdev": 1.0, "random_seed": 1234}),
       None, None, "quad_problem_noise", 5678),
      (_Spec(pg.Rosenbrock, (), {"random_seed": 1234}), None, None,
       "rosenbrock", 5678),
      (_Spec(pg.Rosenbrock, (), {"random_seed": 1234, "noise_stdev": 1.0}),
       None, None, "rosenbrock_noise", 5678),
      (_Spec(pg.SoftmaxRegression, (10, 2), {}), datasets.random(
          10, 10000, random_seed=1234), 100, "softmax", 5678),
      (_Spec(pg.SoftmaxRegression, (10, 2), {"noise_stdev": 1.0}),
       datasets.random(10, 10000, random_seed=1234), 100, "softmax_noise",
       5678),
      (_Spec(pg.FullyConnected, (10, 2), {}), datasets.random(
          10, 10000, random_seed=1234), 100, "mlp_small",
       _test_problem_mlp_scaled_init_small()),
      (_Spec(pg.FullyConnected, (20, 10), {}), datasets.random(
          20, 10000, n_classes=10, random_seed=1234), 100, "mlp_large",
       _test_problem_mlp_scaled_init_large()),
      (_Spec(pg.FullyConnected, (784, 10),
             {"hidden_sizes": (64,), "activation": tf.nn.sigmoid}),
       datasets.mnist(), 64, "mlp_mnist_sigmoid",
       _test_problem_mlp_scaled_init_mnist()),
      (_Spec(pg.FullyConnected, (784, 10),
             {"hidden_sizes": (64,), "activation": tf.nn.relu}),
       datasets.mnist(), 64, "mlp_mnist_relu",
       _test_problem_mlp_scaled_init_mnist()),
      (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
             {"activation": tf.nn.sigmoid}), datasets.mnist(), 64,
       "convnet_mnist_sigmoid", None),
      (_Spec(pg.ConvNet, ((1, 28, 28), 10, [(3, 3, 8), (5, 5, 8)]),
             {"activation": tf.nn.relu}), datasets.mnist(), 64,
       "convnet_mnist_relu", None),
  ]
  return tp
