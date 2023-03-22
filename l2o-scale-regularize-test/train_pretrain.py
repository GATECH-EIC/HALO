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

"""Scripts for train the pre-trained model for adaptation tasks."""

from __future__ import print_function

import os
import pickle

import tensorflow as tf

import metaopt
from optimizer import coordinatewise_rnn
from optimizer import global_learning_rate
from optimizer import hierarchical_rnn
from optimizer import learning_rate_schedule
from optimizer import trainable_adam
from problems import problem_sets as ps
from problems import problem_spec
from problems import datasets


tf.app.flags.DEFINE_string("train_dir", "pretrain/",
                           """Directory to store parameters and results.""")


tf.app.flags.DEFINE_string("test_optimizer", "L2o",
                           """optimizer to test.""")
tf.app.flags.DEFINE_integer("task", 0,
                            """Task id of the replica running the training.""")
tf.app.flags.DEFINE_integer("worker_tasks", 1,
                            """Number of tasks in the worker job.""")
tf.app.flags.DEFINE_integer("num_testing_itrs", 100,
                            """Number testing iterations.""")
tf.app.flags.DEFINE_integer("num_problems", 1,
                            """Number of sub-problems to run.""")
tf.app.flags.DEFINE_integer("num_meta_iterations", 5,
                            """Number of meta-iterations to optimize.""")
tf.app.flags.DEFINE_integer("num_unroll_scale", 10,
                            """The scale parameter of the exponential
                            distribution from which the number of partial
                            unrolls is drawn""")
tf.app.flags.DEFINE_integer("min_num_unrolls", 4,
                            """The minimum number of unrolls per problem.""")
tf.app.flags.DEFINE_integer("num_partial_unroll_itr_scale", 50,
                            """The scale parameter of the exponential
                               distribution from which the number of iterations
                               per unroll is drawn.""")
tf.app.flags.DEFINE_integer("min_num_itr_partial_unroll", 10,
                            """The minimum number of iterations for one
                               unroll.""")

tf.app.flags.DEFINE_string("optimizer", "HierarchicalRNN",
                           """Which meta-optimizer to train.""")

# CoordinatewiseRNN-specific flags
tf.app.flags.DEFINE_integer("cell_size", 10,
                            """Size of the RNN hidden state in each layer.""")
tf.app.flags.DEFINE_integer("num_cells", 2,
                            """Number of RNN layers.""")
tf.app.flags.DEFINE_string("cell_cls", "GRUCell",
                           """Type of RNN cell to use.""")

# Metaoptimization parameters
tf.app.flags.DEFINE_float("meta_learning_rate", 1e-6,
                          """The learning rate for the meta-optimizer.""")
tf.app.flags.DEFINE_float("gradient_clip_level", 1e4,
                          """The level to clip gradients to.""")

# Train or test
# tf.app.flags.DEFINE_boolean("training", False,
#                             """training or testing.""")

# Training set selection
tf.app.flags.DEFINE_boolean("include_mnist_conv_problems", False,
                            """Include Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_TDD_MLP_problems", False,
                            """Include MLP problems.""")
tf.app.flags.DEFINE_boolean("include_PIE_conv_problems", False,
                            """Include MLP problems.""")
tf.app.flags.DEFINE_boolean("include_mnist_conv_problems_wide", False,
                            """Include Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_resnet18", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_vgg", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_vgg11", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_v2", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_v3", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_v4", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_v5", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_v6", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_v7", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems_v8", False,
                            """Include cifar10 Convnet problems.""")

tf.app.flags.DEFINE_boolean("include_MLP_GAS_problems", False,
                            """Include Conv LSTM problems""")


tf.app.flags.DEFINE_float("lr", 0.001,
                          """ learning rate.""")
# Optimizer parameters: initialization and scale values
tf.app.flags.DEFINE_float("min_lr", 1e-6,
                          """The minimum initial learning rate.""")
tf.app.flags.DEFINE_float("max_lr", 1e-2,
                          """The maximum initial learning rate.""")

# Optimizer parameters: small features.
tf.app.flags.DEFINE_boolean("zero_init_lr_weights", True,
                            """Whether to initialize the learning rate weights
                               to 0 rather than the scaled random initialization
                               used for other RNN variables.""")
tf.app.flags.DEFINE_boolean("use_relative_lr", True,
                            """Whether to use the relative learning rate as an
                               input during training. Can only be used if
                               learnable_decay is also True.""")
tf.app.flags.DEFINE_boolean("use_extreme_indicator", False,
                            """Whether to use the extreme indicator for learning
                               rates as an input during training. Can only be
                               used if learnable_decay is also True.""")
tf.app.flags.DEFINE_boolean("use_log_means_squared", True,
                            """Whether to track the log of the mean squared
                               grads instead of the means squared grads.""")
tf.app.flags.DEFINE_boolean("use_problem_lr_mean", True,
                            """Whether to use the mean over all learning rates
                               in the problem when calculating the relative
                               learning rate.""")

# Optimizer parameters: major features
tf.app.flags.DEFINE_boolean("learnable_decay", True,
                            """Whether to learn weights that dynamically
                              modulate the input scale via RMS decay.""")
tf.app.flags.DEFINE_boolean("dynamic_output_scale", True,
                            """Whether to learn weights that dynamically
                               modulate the output scale.""")
tf.app.flags.DEFINE_boolean("use_log_objective", True,
                            """Whether to use the log of the scaled objective
                               rather than just the scaled obj for training.""")
tf.app.flags.DEFINE_boolean("use_attention", False,
                            """Whether to learn where to attend.""")
tf.app.flags.DEFINE_boolean("use_second_derivatives", True,
                            """Whether to use second derivatives.""")
tf.app.flags.DEFINE_integer("num_gradient_scales", 4,
                            """How many different timescales to keep for
                               gradient history. If > 1, also learns a scale
                               factor for gradient history.""")
tf.app.flags.DEFINE_float("max_log_lr", 33,
                          """The maximum log learning rate allowed.""")
tf.app.flags.DEFINE_float("objective_training_max_multiplier", -1,
                          """How much the objective can grow before training on
                             this problem / param pair is terminated. Sets a max
                             on the objective value when multiplied by the
                             initial objective. If <= 0, not used.""")
tf.app.flags.DEFINE_boolean("use_gradient_shortcut", True,
                            """Whether to add a learned affine projection of the
                               gradient to the update delta in addition to the
                               gradient function computed by the RNN.""")
tf.app.flags.DEFINE_boolean("use_lr_shortcut", False,
                            """Whether to add the difference between the current
                               learning rate and the desired learning rate to
                               the RNN input.""")
tf.app.flags.DEFINE_boolean("use_grad_products", True,
                            """Whether to use gradient products in the input to
                               the RNN. Only applicable when num_gradient_scales
                               > 1.""")
tf.app.flags.DEFINE_boolean("use_multiple_scale_decays", False,
                            """Whether to use many-timescale scale decays.""")
tf.app.flags.DEFINE_boolean("use_numerator_epsilon", False,
                            """Whether to use epsilon in the numerator of the
                               log objective.""")
tf.app.flags.DEFINE_boolean("learnable_inp_decay", True,
                            """Whether to learn input decay weight and bias.""")
tf.app.flags.DEFINE_boolean("learnable_rnn_init", True,
                            """Whether to learn RNN state initialization.""")

# Additional tricks
tf.app.flags.DEFINE_boolean("use_output_constrain", False, "Use output constrain on RNN's output.")
tf.app.flags.DEFINE_float("output_constrain_alpha", 0.1, "scale of the LSTM output after adding output constains")
tf.app.flags.DEFINE_string("custom_flag",None, "whatever you want add in the save dir")
FLAGS = tf.app.flags.FLAGS

# The Size of the RNN hidden state in each layer:
# [PerParam, PerTensor, Global]. The length of this list must be 1, 2, or 3.
# If less than 3, the Global and/or PerTensor RNNs will not be created.

if FLAGS.custom_flag is None:
    FLAGS.custom_flag = ""

def main(_):
    """Runs the main script."""


    # Choose a set of problems to optimize. By default this includes quadratics,
    # 2-dimensional bowls, 2-class softmax problems, and non-noisy optimization
    # test problems (e.g. Rosenbrock, Beale)
    problems_and_data = []

    if FLAGS.include_mnist_conv_problems:
        problems_and_data.extend(ps.pretrain_mnist_conv_problems())

    if FLAGS.include_TDD_MLP_problems:
        problems_and_data.extend(ps.pretrain_TDD_MLP_problems())

    if FLAGS.include_PIE_conv_problems:
        problems_and_data.extend(ps.pretrain_PIE_conv_problems())

    if FLAGS.include_mnist_conv_problems_wide:
        problems_and_data.extend(ps.pretrain_mnist_conv_problems_wide())

    if FLAGS.include_cifar10_conv_problems:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems())

    if FLAGS.include_cifar10_resnet18:
        problems_and_data.extend(ps.pretrain_cifar10_resnet18())

    if FLAGS.include_cifar10_conv_problems_vgg:
        problems_and_data.extend(ps.pretrain_vgg16_problems())

    if FLAGS.include_cifar10_conv_problems_vgg11:
        problems_and_data.extend(ps.pretrain_vgg11_problems())

    if FLAGS.include_cifar10_conv_problems_v2:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems_v2())

    if FLAGS.include_cifar10_conv_problems_v3:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems_v3())

    if FLAGS.include_cifar10_conv_problems_v4:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems_v4())

    if FLAGS.include_cifar10_conv_problems_v5:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems_v5())

    if FLAGS.include_cifar10_conv_problems_v6:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems_v6())

    if FLAGS.include_cifar10_conv_problems_v7:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems_v7())

    if FLAGS.include_cifar10_conv_problems_v8:
        problems_and_data.extend(ps.pretrain_cifar10_conv_problems_v8())

    if FLAGS.include_MLP_GAS_problems:
        problems_and_data.extend(ps.pretrain_MLP_GAS_problems())

    # test trainable_optimizer
    opt = None
    if FLAGS.test_optimizer == 'SGD':
      print('using optimzier SGD')
      opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
      logdir = None
    elif FLAGS.test_optimizer == 'Adam':
      print('using optimzier Adam')
      opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
      logdir = None
    elif FLAGS.test_optimizer == 'Adagrad':
      print('using optimzier Adagrad')
      opt = tf.train.AdagradOptimizer(learning_rate=FLAGS.lr)
      logdir = None

    for problem_itr, (problem, dataset, batch_size) in enumerate(problems_and_data):
        # build a new graph for this problem
        print(dataset)
        problem = problem.build()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        problem_name = FLAGS.train_dir.split(os.path.sep)[-2]
        save_dir = os.path.join(current_dir, 'records', problem_name, FLAGS.test_optimizer +"_lr_"+str(FLAGS.lr)+"_"+FLAGS.custom_flag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for seed in range(0, 20, 4):
        # for seed in range(8, 9, 4):
            print('testing the {} using seed {}'.format(problem_name, seed))
            # initialize a problem
            objective_values, acc_records, parameters,_ = metaopt.test_optimizer(
                opt,
                problem,
                num_epochs=FLAGS.num_testing_itrs,
                dataset=dataset,
                batch_size=batch_size,
                seed=seed,
                graph=None,
                logdir=logdir,
                record_every=None,
                include_is_training=True if FLAGS.include_cifar10_resnet18 else False)

            with open(os.path.join(save_dir, 'seed{}_eval_loss_record.pickle'.format(seed)), 'wb') as l1_record:
                pickle.dump(objective_values, l1_record)
            with open(os.path.join(save_dir, 'seed{}_eval_acc_record.pickle'.format(seed)), 'wb') as l2_record:
                pickle.dump(acc_records, l2_record)
            with open(os.path.join(save_dir, 'seed{}_model_params.pickle'.format(seed)), 'wb') as l3_record:
                pickle.dump(parameters, l3_record)
            print("Saving evaluate seed{} loss and acc records to {} ".format(seed, FLAGS.test_optimizer))
    return 0

if __name__ == "__main__":
    tf.app.run()