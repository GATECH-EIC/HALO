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

"""Scripts for meta-optimization."""

from __future__ import print_function

import os
import pickle

import tensorflow as tf

import metaopt_wearable as metaopt
from optimizer import coordinatewise_rnn
from optimizer import global_learning_rate
from optimizer import hierarchical_rnn
from optimizer import learning_rate_schedule
from optimizer import trainable_adam
from problems import problem_sets as ps
from problems import problem_spec
from problems import datasets
import time
import numpy as np

# tf.app.flags.DEFINE_integer("seed", 14,
#                             """random seed.""")

tf.app.flags.DEFINE_string("train_dir", "adapt/",
                           """Directory to store parameters and results.""")

tf.app.flags.DEFINE_string("pretrained_model_path", None,
                           """Directory to store pretrain models weight.""")

tf.app.flags.DEFINE_string("test_optimizer", "L2o",
                           """optimizer to test.""")

tf.app.flags.DEFINE_integer("task", 0,
                            """Task id of the replica running the training.""")
tf.app.flags.DEFINE_integer("worker_tasks", 1,
                            """Number of tasks in the worker job.""")
tf.app.flags.DEFINE_integer("num_testing_itrs", 1,
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
tf.app.flags.DEFINE_boolean("include_quadratic_problems", False,
                            """Include non-noisy quadratic problems.""")
# tf.app.flags.DEFINE_boolean("include_mnist_conv_problems", False,
#                             """Include Convnet problems.""")
# tf.app.flags.DEFINE_boolean("include_cifar10_conv_problems", False,
#                             """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("adapt_cifar10_conv_problems", False,
                            """Include cifar10 Convnet problems.""")
tf.app.flags.DEFINE_boolean("adapt_cifar10_conv_problems_v3", False,
                            """Include cifar10 Convnet problems but deeper.""")
tf.app.flags.DEFINE_boolean("include_vgg16_problems", False,
                            """Include vgg16 Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_mnist_mlp_problems", False,
                            """Include Convnet problems.""")
tf.app.flags.DEFINE_boolean("include_noisy_quadratic_problems", False,
                            """Include noisy quadratic problems.""")
tf.app.flags.DEFINE_boolean("include_large_quadratic_problems", False,
                            """Include very large quadratic problems.""")
tf.app.flags.DEFINE_boolean("include_bowl_problems", False,
                            """Include 2D bowl problems.""")
tf.app.flags.DEFINE_boolean("include_softmax_2_class_problems", False,
                            """Include 2-class logistic regression problems.""")
tf.app.flags.DEFINE_boolean("include_noisy_softmax_2_class_problems", False,
                            """Include noisy 2-class logistic regression
                               problems.""")
tf.app.flags.DEFINE_boolean("include_optimization_test_problems", False,
                            """Include non-noisy versions of classic
                               optimization test problems, e.g. Rosenbrock.""")
tf.app.flags.DEFINE_boolean("include_noisy_optimization_test_problems", False,
                            """Include gradient-noise versions of classic
                               optimization test problems, e.g. Rosenbrock""")
tf.app.flags.DEFINE_boolean("include_fully_connected_random_2_class_problems",
                            False, """Include MLP problems for 2 classes.""")
tf.app.flags.DEFINE_boolean("include_matmul_problems", False,
                            """Include matrix multiplication problems.""")
tf.app.flags.DEFINE_boolean("include_log_objective_problems", False,
                            """Include problems where the objective is the log
                               objective of another problem, e.g. Bowl.""")
tf.app.flags.DEFINE_boolean("include_rescale_problems", False,
                            """Include problems where the parameters are scaled
                               version of the original parameters.""")
tf.app.flags.DEFINE_boolean("include_norm_problems", False,
                            """Include problems where the objective is the
                               N-norm of another problem, e.g. Quadratic.""")
tf.app.flags.DEFINE_boolean("include_sum_problems", False,
                            """Include problems where the objective is the sum
                               of the objectives of the subproblems that make
                               up the problem parameters. Per-problem tensors
                               are still independent of each other.""")
tf.app.flags.DEFINE_boolean("include_sparse_gradient_problems", False,
                            """Include problems where the gradient is set to 0
                               with some high probability.""")
tf.app.flags.DEFINE_boolean("include_sparse_softmax_problems", False,
                            """Include sparse softmax problems.""")
tf.app.flags.DEFINE_boolean("include_one_hot_sparse_softmax_problems", False,
                            """Include one-hot sparse softmax problems.""")
tf.app.flags.DEFINE_boolean("include_noisy_bowl_problems", False,
                            """Include noisy bowl problems.""")
tf.app.flags.DEFINE_boolean("include_noisy_norm_problems", False,
                            """Include noisy norm problems.""")
tf.app.flags.DEFINE_boolean("include_noisy_sum_problems", False,
                            """Include noisy sum problems.""")
tf.app.flags.DEFINE_boolean("include_sum_of_quadratics_problems", False,
                            """Include sum of quadratics problems.""")
tf.app.flags.DEFINE_boolean("include_projection_quadratic_problems", False,
                            """Include projection quadratic problems.""")
tf.app.flags.DEFINE_boolean("include_outward_snake_problems", False,
                            """Include outward snake problems.""")
tf.app.flags.DEFINE_boolean("include_dependency_chain_problems", False,
                            """Include dependency chain problems.""")
tf.app.flags.DEFINE_boolean("include_min_max_well_problems", False,
                            """Include min-max well problems.""")

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
tf.app.flags.DEFINE_string("random_sparse_prob", "1.0", "The probability for random sparse updates, use space between different probs for different layers")
tf.app.flags.DEFINE_string("random_sparse_method", None, "The method for random sparse updates, including layer_wise, params_wise, fix_num")
tf.app.flags.DEFINE_integer("batch_size", 1, "batch size during test optimizer")
tf.app.flags.DEFINE_float("lr", 0.001, "learning rate for Adam/Adagrad/SGD")
FLAGS = tf.app.flags.FLAGS

# The Size of the RNN hidden state in each layer:
# [PerParam, PerTensor, Global]. The length of this list must be 1, 2, or 3.
# If less than 3, the Global and/or PerTensor RNNs will not be created.

HRNN_CELL_SIZES = [10, 20, 20]


def register_optimizers():
    opts = {}
    opts["CoordinatewiseRNN"] = coordinatewise_rnn.CoordinatewiseRNN
    opts["GlobalLearningRate"] = global_learning_rate.GlobalLearningRate
    opts["HierarchicalRNN"] = hierarchical_rnn.HierarchicalRNN
    opts["LearningRateSchedule"] = learning_rate_schedule.LearningRateSchedule
    opts["TrainableAdam"] = trainable_adam.TrainableAdam
    return opts


def main(_):
    """Runs the main script."""
    
    # convert random_sparse_prob to list
    prob_str_list = FLAGS.random_sparse_prob.split(" ")
    prob_float_list = [float(s) for s in prob_str_list]
    # this is used to write to the dir path
    prob_dir_list = ""
    for fs in prob_str_list:
        prob_dir_list = prob_dir_list + "_" + fs

    print("prob_float_list: ", prob_float_list)
    print("Using method: ", FLAGS.random_sparse_method)

    opts = register_optimizers()
    
    # Choose a set of problems to optimize. By default this includes quadratics,
    # 2-dimensional bowls, 2-class softmax problems, and non-noisy optimization
    # test problems (e.g. Rosenbrock, Beale)
    problems_and_data = []
    
    if FLAGS.include_sparse_softmax_problems:
        problems_and_data.extend(ps.sparse_softmax_2_class_sparse_problems())

    # if FLAGS.include_mnist_conv_problems:
    #     problems_and_data.extend(ps.test_mnist_conv_problems())
    
    # if FLAGS.include_cifar10_conv_problems:
    #     problems_and_data.extend(ps.test_cifar10_conv_problems())
    
    if FLAGS.adapt_cifar10_conv_problems:
        problems_and_data.extend(ps.adapt_cifar10_conv_problems(batch_size=FLAGS.batch_size, random_sparse_method=FLAGS.random_sparse_method, random_sparse_prob=prob_float_list))

    if FLAGS.adapt_cifar10_conv_problems_v3:
        problems_and_data.extend(ps.adapt_cifar10_conv_problems_v3(batch_size=FLAGS.batch_size, random_sparse_method=FLAGS.random_sparse_method, random_sparse_prob=prob_float_list))

    if FLAGS.include_vgg16_problems:
        problems_and_data.extend(ps.vgg16_problems())
    
    if FLAGS.include_mnist_mlp_problems:
        problems_and_data.extend(ps.mnist_mlp_problems())
    
    if FLAGS.include_one_hot_sparse_softmax_problems:
        problems_and_data.extend(
            ps.one_hot_sparse_softmax_2_class_sparse_problems())
    
    if FLAGS.include_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems())
    
    if FLAGS.include_noisy_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems_noisy())
    
    if FLAGS.include_large_quadratic_problems:
        problems_and_data.extend(ps.quadratic_problems_large())
    
    if FLAGS.include_bowl_problems:
        problems_and_data.extend(ps.bowl_problems())
    
    if FLAGS.include_noisy_bowl_problems:
        problems_and_data.extend(ps.bowl_problems_noisy())
    
    if FLAGS.include_softmax_2_class_problems:
        problems_and_data.extend(ps.softmax_2_class_problems())
    
    if FLAGS.include_noisy_softmax_2_class_problems:
        problems_and_data.extend(ps.softmax_2_class_problems_noisy())
    
    if FLAGS.include_optimization_test_problems:
        problems_and_data.extend(ps.optimization_test_problems())
    
    if FLAGS.include_noisy_optimization_test_problems:
        problems_and_data.extend(ps.optimization_test_problems_noisy())
    
    if FLAGS.include_fully_connected_random_2_class_problems:
        problems_and_data.extend(ps.fully_connected_random_2_class_problems())
    
    if FLAGS.include_matmul_problems:
        problems_and_data.extend(ps.matmul_problems())
    
    if FLAGS.include_log_objective_problems:
        problems_and_data.extend(ps.log_objective_problems())
    
    if FLAGS.include_rescale_problems:
        problems_and_data.extend(ps.rescale_problems())
    
    if FLAGS.include_norm_problems:
        problems_and_data.extend(ps.norm_problems())
    
    if FLAGS.include_noisy_norm_problems:
        problems_and_data.extend(ps.norm_problems_noisy())
    
    if FLAGS.include_sum_problems:
        problems_and_data.extend(ps.sum_problems())
    
    if FLAGS.include_noisy_sum_problems:
        problems_and_data.extend(ps.sum_problems_noisy())
    
    if FLAGS.include_sparse_gradient_problems:
        problems_and_data.extend(ps.sparse_gradient_problems())
        if FLAGS.include_fully_connected_random_2_class_problems:
            problems_and_data.extend(ps.sparse_gradient_problems_mlp())
    
    if FLAGS.include_min_max_well_problems:
        problems_and_data.extend(ps.min_max_well_problems())
    
    if FLAGS.include_sum_of_quadratics_problems:
        problems_and_data.extend(ps.sum_of_quadratics_problems())
    
    if FLAGS.include_projection_quadratic_problems:
        problems_and_data.extend(ps.projection_quadratic_problems())
    
    if FLAGS.include_outward_snake_problems:
        problems_and_data.extend(ps.outward_snake_problems())
    
    if FLAGS.include_dependency_chain_problems:
        problems_and_data.extend(ps.dependency_chain_problems())
    
    # log directory
    # logdir = os.path.join(FLAGS.train_dir,
    #                       "{}_{}_{}_{}".format(FLAGS.optimizer,
    #                                            FLAGS.cell_cls,
    #                                            FLAGS.cell_size,
    #                                            FLAGS.num_cells))
    logdir = os.path.join(FLAGS.train_dir,
                            "{}_{}_{}_{}_{}{}".format(FLAGS.optimizer,
                                                FLAGS.cell_cls,
                                                FLAGS.cell_size,
                                                FLAGS.num_cells,
                                                FLAGS.random_sparse_method,
                                                prob_dir_list))
    print("logdir", logdir)

    # get the optimizer class and arguments
    optimizer_cls = opts[FLAGS.optimizer]
    
    assert len(HRNN_CELL_SIZES) in [1, 2, 3]
    optimizer_args = (HRNN_CELL_SIZES,)
    
    optimizer_kwargs = {
        "init_lr_range": (FLAGS.min_lr, FLAGS.max_lr),
        "learnable_decay": FLAGS.learnable_decay,
        "dynamic_output_scale": FLAGS.dynamic_output_scale,
        "cell_cls": getattr(tf.contrib.rnn, FLAGS.cell_cls),
        "use_attention": FLAGS.use_attention,
        "use_log_objective": FLAGS.use_log_objective,
        "num_gradient_scales": FLAGS.num_gradient_scales,
        "zero_init_lr_weights": FLAGS.zero_init_lr_weights,
        "use_log_means_squared": FLAGS.use_log_means_squared,
        "use_relative_lr": FLAGS.use_relative_lr,
        "use_extreme_indicator": FLAGS.use_extreme_indicator,
        "max_log_lr": FLAGS.max_log_lr,
        "obj_train_max_multiplier": FLAGS.objective_training_max_multiplier,
        "use_problem_lr_mean": FLAGS.use_problem_lr_mean,
        "use_gradient_shortcut": FLAGS.use_gradient_shortcut,
        "use_second_derivatives": FLAGS.use_second_derivatives,
        "use_lr_shortcut": FLAGS.use_lr_shortcut,
        "use_grad_products": FLAGS.use_grad_products,
        "use_multiple_scale_decays": FLAGS.use_multiple_scale_decays,
        "use_numerator_epsilon": FLAGS.use_numerator_epsilon,
        "learnable_inp_decay": FLAGS.learnable_inp_decay,
        "learnable_rnn_init": FLAGS.learnable_rnn_init,
        "use_output_constrain": FLAGS.use_output_constrain,
        "output_constrain_alpha": FLAGS.output_constrain_alpha,
        "random_sparse_method": FLAGS.random_sparse_method if FLAGS.random_sparse_method is not None else "layer_wise",
        "random_sparse_prob": prob_float_list
    }
    
    
    # make log directory
    tf.gfile.MakeDirs(logdir)
    
    is_chief = FLAGS.task == 0
    # if this is a distributed run, make the chief run through problems in order
    select_random_problems = FLAGS.worker_tasks == 1 or not is_chief
    
    def num_unrolls():
        return metaopt.sample_numiter(FLAGS.num_unroll_scale, FLAGS.min_num_unrolls)
    
    def num_partial_unroll_itrs():
        return metaopt.sample_numiter(FLAGS.num_partial_unroll_itr_scale,
                                      FLAGS.min_num_itr_partial_unroll)
    
    
    # test trainable_optimizer
    opt = None
    if FLAGS.test_optimizer == 'L2o':
      print('using optimzier L2o')
      optimizer_spec = problem_spec.Spec(
            optimizer_cls, optimizer_args, optimizer_kwargs)
      opt = optimizer_spec.build()
      logdir = logdir
    elif FLAGS.test_optimizer == 'SGD':
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
    # if dataset is None, use the EMPTY_DATASET
    # if dataset is None:
    #   dataset = datasets.EMPTY_DATASET
    #   batch_size = dataset.size
    for problem_itr, (problem, dataset, batch_size) in enumerate(problems_and_data):
        # build a new graph for this problem
        print(dataset)
        problem = problem.build()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        problem_name = FLAGS.train_dir.split(os.path.sep)[-2]
        problem_feature = FLAGS.train_dir.split(os.path.sep)[-1]
        
        if FLAGS.test_optimizer != 'L2o':
            lr_str = "_lr_"+str(FLAGS.lr)
        else:
            lr_str = ""

        if FLAGS.random_sparse_method is not None:
            save_dir = os.path.join(current_dir, 'records_wearable', problem_name, FLAGS.test_optimizer +"_"+problem_feature+"_"+str(FLAGS.batch_size)+"_"+FLAGS.random_sparse_method+prob_dir_list+lr_str)
        else:
            save_dir = os.path.join(current_dir, 'records_wearable', problem_name, FLAGS.test_optimizer +"_"+problem_feature+"_"+str(FLAGS.batch_size)+lr_str)
        # save_dir = os.path.join(current_dir, 'tmd', problem_name, FLAGS.test_optimizer +"_"+problem_feature)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        time_list = []
        for seed in range(0, 20, 2):
            print('testing the {} using seed {}'.format(problem_name, seed))
            # initialize a problem
            
            objective_values, acc_records, parameters, time_d = metaopt.test_optimizer(
                opt,
                problem,
                num_epochs=FLAGS.num_testing_itrs,
                dataset=dataset,
                batch_size=batch_size,
                seed=seed,
                graph=None,
                logdir=logdir,
                record_every=None,
                pretrained_model_path=FLAGS.pretrained_model_path)

            time_list.append(time_d)
            with open(os.path.join(save_dir, 'seed{}_eval_loss_record.pickle'.format(seed)), 'wb') as l1_record:
                pickle.dump(objective_values, l1_record)
            with open(os.path.join(save_dir, 'seed{}_eval_acc_record.pickle'.format(seed)), 'wb') as l2_record:
                pickle.dump(acc_records, l2_record)
            with open(os.path.join(save_dir, 'seed{}_model_params.pickle'.format(seed)), 'wb') as l3_record:
                pickle.dump(parameters, l3_record)
            print("Saving evaluate seed{} loss and acc records to {} ".format(seed, FLAGS.test_optimizer))
            print("Running {} epochs using {} s".format(FLAGS.num_testing_itrs, time_d))
        print("Average Running {} epochs using {} s".format(FLAGS.num_testing_itrs, np.array(time_list).mean()))
    return 0


if __name__ == "__main__":
    tf.app.run()