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

"""Collection of trainable optimizers for meta-optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import state_ops
from optimizer import rnn_cells
from optimizer import trainable_optimizer as opt
from optimizer import utils
import itertools

# Default was 0.1
tf.app.flags.DEFINE_float("biasgrucell_scale", 0.5,
                          """The scale for the internal BiasGRUCell vars.""")
# Default was 0
tf.app.flags.DEFINE_float("biasgrucell_gate_bias_init", 2.2,
                          """The bias for the internal BiasGRUCell reset and
                             update gate variables.""")
# Default was 1e-3
tf.app.flags.DEFINE_float("hrnn_rnn_readout_scale", 0.5,
                          """The initialization scale for the RNN readouts.""")
tf.app.flags.DEFINE_float("hrnn_default_decay_var_init", 2.2,
                          """The default initializer value for any decay/
                             momentum style variables and constants.
                             sigmoid(2.2) ~ 0.9, sigmoid(-2.2) ~ 0.01.""")
# Default was 2.2
tf.app.flags.DEFINE_float("scale_decay_bias_init", 3.2,
                          """The initialization for the scale decay bias. This
                             is the initial bias for the timescale for the
                             exponential avg of the mean square gradients.""")
tf.app.flags.DEFINE_float("learning_rate_momentum_logit_init", 3.2,
                          """Initialization for the learning rate momentum.""")
# Default was 0.1
tf.app.flags.DEFINE_float("hrnn_affine_scale", 0.5,
                          """The initialization scale for the weight matrix of
                             the bias variables in layer0 and 1 of the hrnn.""")

FLAGS = tf.flags.FLAGS

# Function to combine updates and no_updates variables together and reshape it the corresponding shape
# Notice that the indices is the axis = 0
def combine_updates(var_updates, var_updates_indices, var_no_updates, var_no_updates_indices, var_shape):
  var_combine = tf.concat([var_updates, var_no_updates], 0)
  indices_combine = tf.concat([var_updates_indices, var_no_updates_indices], 0)
  indices_reverse = tf.argsort(indices_combine)
  var_combine_sort = tf.gather(var_combine, indices_reverse)
  return tf.reshape(var_combine_sort, var_shape)

class HierarchicalRNN(opt.TrainableOptimizer):
  """3 level hierarchical RNN.

  Optionally uses second order gradient information and has decoupled evaluation
  and update locations.
  """

  def __init__(self, level_sizes, init_lr_range=(1e-6, 1e-2),
               learnable_decay=True, dynamic_output_scale=True,
               use_attention=False, use_log_objective=True,
               num_gradient_scales=4, zero_init_lr_weights=True,
               use_log_means_squared=True, use_relative_lr=True,
               use_extreme_indicator=False, max_log_lr=33,
               obj_train_max_multiplier=-1, use_problem_lr_mean=False,
               use_gradient_shortcut=False, use_lr_shortcut=False,
               use_grad_products=False, use_multiple_scale_decays=False,
               learnable_inp_decay=True, learnable_rnn_init=True,
               random_seed=None, use_output_constrain=False,
               output_constrain_alpha=0.1, random_sparse_method="layer_wise",
               random_sparse_prob=[1.0,], **kwargs):
    """Initializes the RNN per-parameter optimizer.

    The hierarchy consists of up to three levels:
    Level 0: per parameter RNN
    Level 1: per tensor RNN
    Level 2: global RNN

    Args:
      level_sizes: list or tuple with 1, 2, or 3 integers, the number of units
          in each RNN in the hierarchy (level0, level1, level2).
          length 1: only coordinatewise rnn's will be used
          length 2: coordinatewise and tensor-level rnn's will be used
          length 3: a single global-level rnn will be used in addition to
             coordinatewise and tensor-level
      init_lr_range: the range in which to initialize the learning rates
      learnable_decay: whether to learn weights that dynamically modulate the
          input scale via RMS style decay
      dynamic_output_scale: whether to learn weights that dynamically modulate
          the output scale
      use_attention: whether to use attention to train the optimizer
      use_log_objective: whether to train on the log of the objective
      num_gradient_scales: the number of scales to use for gradient history
      zero_init_lr_weights: whether to initialize the lr weights to zero
      use_log_means_squared: whether to track the log of the means_squared,
          used as a measure of signal vs. noise in gradient.
      use_relative_lr: whether to use the relative learning rate as an
          input during training (requires learnable_decay=True)
      use_extreme_indicator: whether to use the extreme indicator for learning
          rates as an input during training (requires learnable_decay=True)
      max_log_lr: the maximum log learning rate allowed during train or test
      obj_train_max_multiplier: max objective increase during a training run
      use_problem_lr_mean: whether to use the mean over all learning rates in
          the problem when calculating the relative learning rate as opposed to
          the per-tensor mean
      use_gradient_shortcut: Whether to add a learned affine projection of the
          gradient to the update delta in addition to the gradient function
          computed by the RNN
      use_lr_shortcut: Whether to add as input the difference between the log lr
          and the desired log lr (1e-3)
      use_grad_products: Whether to use gradient products in the rnn input.
          Only applicable if num_gradient_scales > 1
      use_multiple_scale_decays: Whether to use multiple scales for the scale
          decay, as with input decay
      learnable_inp_decay: Whether to learn the input decay weights and bias.
      learnable_rnn_init: Whether to learn the RNN state initialization.
      random_seed: Random seed for random variable initializers. (Default: None)
      **kwargs: args passed to TrainableOptimizer's constructor

    Raises:
      ValueError: If level_sizes is not a length 1, 2, or 3 list.
      ValueError: If there are any non-integer sizes in level_sizes.
      ValueError: If the init lr range is not of length 2.
      ValueError: If the init lr range is not a valid range (min > max).
    """
    if len(level_sizes) not in [1, 2, 3]:
      raise ValueError("HierarchicalRNN only supports 1, 2, or 3 levels in the "
                       "hierarchy, but {} were requested.".format(
                           len(level_sizes)))
    if any(not isinstance(level, int) for level in level_sizes):
      raise ValueError("Level sizes must be integer values, were {}".format(
          level_sizes))
    if len(init_lr_range) != 2:
      raise ValueError(
          "Initial LR range must be len 2, was {}".format(len(init_lr_range)))
    if init_lr_range[0] > init_lr_range[1]:
      raise ValueError("Initial LR range min is greater than max.")

    self.learnable_decay = learnable_decay
    self.dynamic_output_scale = dynamic_output_scale
    self.use_attention = use_attention
    self.use_log_objective = use_log_objective
    self.num_gradient_scales = num_gradient_scales
    self.zero_init_lr_weights = zero_init_lr_weights
    self.use_log_means_squared = use_log_means_squared
    self.use_relative_lr = use_relative_lr
    self.use_extreme_indicator = use_extreme_indicator
    self.max_log_lr = max_log_lr
    self.use_problem_lr_mean = use_problem_lr_mean
    self.use_gradient_shortcut = use_gradient_shortcut
    self.use_lr_shortcut = use_lr_shortcut
    self.use_grad_products = use_grad_products
    self.use_multiple_scale_decays = use_multiple_scale_decays
    self.learnable_inp_decay = learnable_inp_decay
    self.learnable_rnn_init = learnable_rnn_init

    self.use_output_constrain = use_output_constrain
    self.output_constrain_alpha = output_constrain_alpha

    self.random_sparse_prob = random_sparse_prob
    self.random_sparse_method = random_sparse_method

    self.random_seed = random_seed

    self.num_layers = len(level_sizes)
    self.init_lr_range = init_lr_range

    self.reuse_vars = None
    self.reuse_global_state = None
    self.cells = []
    self.init_vectors = []

    with tf.variable_scope(opt.OPTIMIZER_SCOPE):

      self._initialize_rnn_cells(level_sizes)

      # get the cell size for the per-parameter RNN (Level 0)
      cell_size = level_sizes[0]

      # Random normal initialization scaled by the output size. This is the
      # scale for the RNN *readouts*. RNN internal weight scale is set in the
      # BiasGRUCell call.
      scale_factor = FLAGS.hrnn_rnn_readout_scale / math.sqrt(cell_size)
      scaled_init = tf.random_normal_initializer(0., scale_factor,
                                                 seed=self.random_seed)

      # weights for projecting the hidden state to a parameter update
      self.update_weights = tf.get_variable("update_weights",
                                            shape=(cell_size, 1),
                                            initializer=scaled_init)

      if self.use_attention:
        # weights for projecting the hidden state to the location at which the
        # gradient is attended
        self.attention_weights = tf.get_variable(
            "attention_weights",
            initializer=self.update_weights.initialized_value())

      # weights for projecting the hidden state to the RMS decay term
      self._initialize_scale_decay((cell_size, 1), scaled_init)
      self._initialize_input_decay((cell_size, 1), scaled_init)

      self._initialize_lr((cell_size, 1), scaled_init)

    state_keys = ["parameter", "layer", "scl_decay", "inp_decay", "true_param"]

    if self.dynamic_output_scale:
      state_keys.append("log_learning_rate")

    for i in range(self.num_gradient_scales):
      state_keys.append("grad_accum{}".format(i + 1))
      state_keys.append("ms{}".format(i + 1))

    super(HierarchicalRNN, self).__init__(
        "hRNN", state_keys, use_attention=use_attention,
        use_log_objective=use_log_objective,
        obj_train_max_multiplier=obj_train_max_multiplier, **kwargs)

  def _initialize_rnn_cells(self, level_sizes):
    """Initializes the RNN cells to use in the hierarchical RNN."""

    # RNN Cell layers (0 -> lowest, 1 -> middle, 2 -> global)
    for level in range(self.num_layers):
      scope = "Level{}_RNN".format(level)
      with tf.variable_scope(scope):
        hcell = rnn_cells.BiasGRUCell(
            level_sizes[level],
            scale=FLAGS.biasgrucell_scale,
            gate_bias_init=FLAGS.biasgrucell_gate_bias_init,
            random_seed=self.random_seed)
        self.cells.append(hcell)
        if self.learnable_rnn_init:
          self.init_vectors.append(tf.Variable(
              tf.random_uniform([1, hcell.state_size], -1., 1.,
                                seed=self.random_seed),
              name="init_vector"))
        else:
          self.init_vectors.append(
              tf.random_uniform([1, hcell.state_size], -1., 1.,
                                seed=self.random_seed))

  def _initialize_scale_decay(self, weights_tensor_shape, scaled_init):
    """Initializes the scale decay weights and bias variables or tensors.

    Args:
      weights_tensor_shape: The shape the weight tensor should take.
      scaled_init: The scaled initialization for the weights tensor.
    """
    if self.learnable_decay:
      self.scl_decay_weights = tf.get_variable("scl_decay_weights",
                                               shape=weights_tensor_shape,
                                               initializer=scaled_init)
      scl_decay_bias_init = tf.constant_initializer(
          FLAGS.scale_decay_bias_init)
      self.scl_decay_bias = tf.get_variable("scl_decay_bias",
                                            shape=(1,),
                                            initializer=scl_decay_bias_init)
    else:
      self.scl_decay_weights = tf.zeros_like(self.update_weights)
      self.scl_decay_bias = tf.log(0.93 / (1. - 0.93))

  def _initialize_input_decay(self, weights_tensor_shape, scaled_init):
    """Initializes the input scale decay weights and bias variables or tensors.

    Args:
      weights_tensor_shape: The shape the weight tensor should take.
      scaled_init: The scaled initialization for the weights tensor.
    """
    if (self.learnable_decay and self.num_gradient_scales > 1 and
        self.learnable_inp_decay):
      self.inp_decay_weights = tf.get_variable("inp_decay_weights",
                                               shape=weights_tensor_shape,
                                               initializer=scaled_init)
      inp_decay_bias_init = tf.constant_initializer(
          FLAGS.hrnn_default_decay_var_init)
      self.inp_decay_bias = tf.get_variable("inp_decay_bias",
                                            shape=(1,),
                                            initializer=inp_decay_bias_init)
    else:
      self.inp_decay_weights = tf.zeros_like(self.update_weights)
      self.inp_decay_bias = tf.log(0.89 / (1. - 0.89))

  def _initialize_lr(self, weights_tensor_shape, scaled_init):
    """Initializes the learning rate weights and bias variables or tensors.

    Args:
      weights_tensor_shape: The shape the weight tensor should take.
      scaled_init: The scaled initialization for the weights tensor.
    """
    if self.dynamic_output_scale:
      zero_init = tf.constant_initializer(0.)
      wt_init = zero_init if self.zero_init_lr_weights else scaled_init
      self.lr_weights = tf.get_variable("learning_rate_weights",
                                        shape=weights_tensor_shape,
                                        initializer=wt_init)
      self.lr_bias = tf.get_variable("learning_rate_bias", shape=(1,),
                                     initializer=zero_init)
    else:
      self.lr_weights = tf.zeros_like(self.update_weights)
      self.lr_bias = tf.zeros([1, 1])

  def _initialize_state(self, var):
    """Return a dictionary mapping names of state variables to their values."""
    var_vectorized = tf.reshape(var, [-1, 1])
    ndim = var_vectorized.get_shape().as_list()[0]

    state = {
        # parameter init tensor is [var_ndim x layer0_cell_size]
        "parameter": tf.ones([ndim, 1]) * self.init_vectors[0],
        "scl_decay": tf.zeros_like(var_vectorized),
        "inp_decay": tf.zeros_like(var_vectorized),
        "true_param": var,
    }

    if self.num_layers > 1:
      # layer init tensor is [1 x layer1_cell_size]
      state["layer"] = tf.ones([1, 1]) * self.init_vectors[1]

    if self.dynamic_output_scale:
      min_lr = self.init_lr_range[0]
      max_lr = self.init_lr_range[1]
      if min_lr == max_lr:
        log_init_lr = tf.log(min_lr * tf.ones_like(var_vectorized))
      else:
        # Use a random offset to increase the likelihood that the average of the
        # LRs for this variable is different from the LRs for other variables.
        actual_vals = tf.random_uniform(var_vectorized.get_shape().as_list(),
                                        np.log(min_lr) / 2.,
                                        np.log(max_lr) / 2.,
                                        seed=self.random_seed)
        offset = tf.random_uniform((), np.log(min_lr) / 2., np.log(max_lr) / 2.,
                                   seed=self.random_seed)
        log_init_lr = actual_vals + offset
      # Clip the log learning rate to the flag at the top end, and to
      # (log(min int32) - 1) at the bottom
      clipped = tf.clip_by_value(log_init_lr, -33, self.max_log_lr)
      state["log_learning_rate"] = clipped

    for i in range(self.num_gradient_scales):
      state["grad_accum{}".format(i + 1)] = tf.zeros_like(var_vectorized)
      state["ms{}".format(i + 1)] = tf.zeros_like(var_vectorized)

    return state

  def _initialize_global_state(self):
    if self.num_layers < 3:
      return []
    rnn_global_init = tf.ones([1, 1]) * self.init_vectors[2]
    return [rnn_global_init]

  def _compute_updates(self, params, grads, states, global_state):
    # Store the updated parameters and states.
    updated_params = []
    updated_attention = []
    updated_states = []

    with tf.variable_scope(opt.OPTIMIZER_SCOPE):

      mean_log_lr = self._compute_mean_log_lr(states)

      # Iterate over the layers.
      # for param, grad_unflat, state in zip(params, grads, states):
      # use cycle for the zip here
      # refer: https://discuss.codecademy.com/t/does-the-zip-function-work-with-lists-of-different-lengths/351451
      for _param, _grad_unflat, _state, _rd_ratio in zip(params, grads, states, itertools.cycle(self.random_sparse_prob)):
        with tf.variable_scope("PerTensor", reuse=tf.AUTO_REUSE):
          # param_shape will record the original shape of params
          param_shape = _param.shape

          self.reuse_vars = True
          _grad = tf.reshape(_grad_unflat, [-1, 1])
          _param_flat = tf.reshape(_param, [-1, 1])

          # flat_shape will record the number of params, which will be used in variables in state
          flat_shape = [_param_flat.shape[0], -1]
          parameter_flat_shape = _state["parameter"].shape
          scl_decay_flat_shape = _state["scl_decay"].shape
          inp_decay_flat_shape = _state["inp_decay"].shape
          log_learning_rate_flat_shape = _state["log_learning_rate"].shape
          grad_accum_flat_shape = _state["grad_accum1"].shape
          ms_flat_shape = _state["ms1"].shape

          assert _grad.shape == _param_flat.shape, ("Shape mismatch between grad and param after flat")
          # let's consider layer-wise spare update first
          if self.random_sparse_method == "layer_wise":
            rd = tf.random.uniform(shape=[1,], maxval=1)
            rd_bool = tf.math.less_equal(rd, _rd_ratio)
            rd_bool = tf.tile(rd_bool, [_param_flat.shape[0]])
            rd_bool_not = tf.math.logical_not(rd_bool)
          elif self.random_sparse_method == "params_wise":
            rd = tf.random.uniform(shape=[int(_param_flat.shape[0]),], maxval=1)
            rd_bool = tf.math.less_equal(rd, _rd_ratio)
            rd_bool_not = tf.math.logical_not(rd_bool)
          elif self.random_sparse_method == "fix_num":
            total_num = int(_param_flat.shape[0])
            select_num = int(round(total_num*_rd_ratio))
            rd = tf.random.uniform(shape=[total_num,], maxval=1)
            _, indices = tf.nn.top_k(rd, select_num)
            scatter = tf.scatter_nd(tf.reshape(indices, [-1,1]), tf.ones(indices.shape), tf.constant([total_num]))
            rd_bool = tf.cast(scatter, tf.bool)
            rd_bool_not = tf.math.logical_not(rd_bool)

          rd_indices = tf.squeeze(tf.where(rd_bool),1)
          rd_indices_not = tf.squeeze(tf.where(rd_bool_not),1)

          param = tf.gather(_param_flat, rd_indices)      
          param_no_update = tf.gather(_param_flat, rd_indices_not)

          grad = tf.gather(_grad, rd_indices)
          grad_no_update = tf.gather(_grad, rd_indices_not)

          state_no_update = {}
          state = {}
          for k in _state:
            if _state[k].shape[0] == _grad.shape[0]:
              state_rd_indices = rd_indices
              state_rd_indices_not = rd_indices_not

              state[k] = tf.gather(_state[k], state_rd_indices)
              state_no_update[k] = tf.gather(_state[k], state_rd_indices_not)
            # for true params in state
            elif len(_state[k].shape) > 3:
              true_param_flat = tf.reshape(_state[k], [-1, 1])
              state_rd_indices = rd_indices
              state_rd_indices_not = rd_indices_not

              state[k] = tf.gather(true_param_flat, state_rd_indices)
              state_no_update[k] = tf.gather(true_param_flat, state_rd_indices_not)
            else:
              # for "layer" in state, the shape is always (1, 20)
              state[k] = _state[k]
              state_no_update[k] = None
          
          # define the updates as functions and use lambda to calls the function
          # refer: https://github.com/tensorflow/tensorflow/issues/3287
          def copy_previous():
            new_state_list = []
            new_state_list.append(_state["parameter"])
            new_state_list.append(_state["scl_decay"])
            new_state_list.append(_state["inp_decay"])
            new_state_list.append(_param)

            if _state["layer"] is not None:
              new_state_list.append(_state["layer"])
              self.layer_state_exist = True
            else:
              self.layer_state_exist = False

            if self.dynamic_output_scale:
              new_state_list.append(_state["log_learning_rate"])

            for i in range(self.num_gradient_scales):
              new_state_list.append(_state["grad_accum{}".format(i + 1)])
              new_state_list.append(_state["ms{}".format(i + 1)])

            return [_param, _param] + new_state_list

          def run_update():
            # Create the RNN input. We will optionally extend it with additional
            # features such as curvature and gradient signal vs. noise.
            (grads_scaled, mean_squared_gradients,
            grads_accum) = self._compute_scaled_and_ms_grads(grad, state)
            rnn_input = [g for g in grads_scaled]

            self._extend_rnn_input(rnn_input, state, grads_scaled,
                                  mean_squared_gradients, mean_log_lr)

            # Concatenate any features we've collected.
            rnn_input_tensor = tf.concat(rnn_input, 1)

            # global state is a list of variabled in (1, 20)
            layer_state, new_param_state = self._update_rnn_cells(
                state, global_state, rnn_input_tensor,
                len(rnn_input) != len(grads_scaled))

            (scl_decay, inp_decay, new_log_lr, update_step, lr_attend,
            attention_delta) = self._compute_rnn_state_projections(
                state, new_param_state, grads_scaled)

            # notice, the update here is still 1D
            # apply output constrain to bound the effective step size
            if self.use_output_constrain:
              update_step = self.output_constrain_alpha*tf.math.tanh(update_step)
            # Apply updates and store state variables.
            if self.use_attention:
              truth = state["true_param"]
              updated_param = truth - update_step
              # convert updated_param
              updated_param = combine_updates(updated_param, rd_indices, 
                              state_no_update["true_param"], rd_indices_not, 
                              param_shape)
              attention_step = tf.reshape(lr_attend * attention_delta,
                                          truth.get_shape())
              # convert updated_attention
              attention_to_append = truth - attention_step
              attention_to_append = combine_updates(attention_to_append, rd_indices, 
                              state_no_update["true_param"], rd_indices_not, 
                              param_shape)

            else:
              updated_param = param - update_step
              # convert updated_param
              updated_param = combine_updates(updated_param, rd_indices, 
                              param_no_update, rd_indices_not, 
                              param_shape)

              attention_to_append = updated_param

            # convert parameter in state
            new_param_state = combine_updates(new_param_state, rd_indices, 
                  state_no_update["parameter"], rd_indices_not, 
                  parameter_flat_shape)
            # convert scl_decay in state
            scl_decay = combine_updates(scl_decay, rd_indices, 
                  state_no_update["scl_decay"], rd_indices_not, 
                  scl_decay_flat_shape)
            # convert inp_decay in state
            inp_decay = combine_updates(inp_decay, rd_indices, 
                  state_no_update["inp_decay"], rd_indices_not, 
                  inp_decay_flat_shape)

            new_state_list = []
            new_state_list.append(new_param_state)
            new_state_list.append(scl_decay)
            new_state_list.append(inp_decay)
            new_state_list.append(updated_param)

            if layer_state is not None:
              new_state_list.append(layer_state)
              self.layer_state_exist = True
            else:
              self.layer_state_exist = False

            # convert log_learning_rate in state
            new_log_lr = combine_updates(new_log_lr, rd_indices, 
                  state_no_update["log_learning_rate"], rd_indices_not, 
                  log_learning_rate_flat_shape)
            if self.dynamic_output_scale:
              new_state_list.append(new_log_lr)

            for i in range(self.num_gradient_scales):
              # convert grad_accum in state
              grads_accum[i] = combine_updates(grads_accum[i], rd_indices, 
                state_no_update["grad_accum{}".format(i + 1)], rd_indices_not, 
                grad_accum_flat_shape)
              # convert ms in state
              mean_squared_gradients[i] = combine_updates(mean_squared_gradients[i], rd_indices, 
                state_no_update["ms{}".format(i + 1)], rd_indices_not, 
                ms_flat_shape)
              new_state_list.append(grads_accum[i])
              new_state_list.append(mean_squared_gradients[i])

            return [updated_param, attention_to_append] + new_state_list
          
          # have to use tf.shape here, refer:
          # https://stackoverflow.com/questions/40685087/tensorflow-converting-unknown-dimension-size-of-a-tensor-to-int
          final_tensor_list = tf.cond(tf.equal(tf.shape(param)[0], tf.constant(0)), lambda: copy_previous(), lambda: run_update())

          final_updated_param = final_tensor_list[0]
          updated_params.append(final_updated_param)

          final_attention_to_append = final_tensor_list[1]
          updated_attention.append(final_attention_to_append)

          final_new_state_list = final_tensor_list[2:]    
          final_new_state = {
            "parameter": final_new_state_list[0],
            "scl_decay": final_new_state_list[1],
            "inp_decay": final_new_state_list[2],
            "true_param": final_new_state_list[3],
          }

          loop_idx_start = 4
          if self.layer_state_exist:
            final_new_state["layer"] = final_new_state_list[loop_idx_start]
            loop_idx_start += 1
          if self.dynamic_output_scale:
            final_new_state["log_learning_rate"] = final_new_state_list[loop_idx_start]
            loop_idx_start += 1
          for i in range(self.num_gradient_scales):
            final_new_state["grad_accum{}".format(i + 1)] = final_new_state_list[loop_idx_start]
            loop_idx_start += 1
            final_new_state["ms{}".format(i + 1)] = final_new_state_list[loop_idx_start]
            loop_idx_start += 1
          updated_states.append(final_new_state)
          
      updated_global_state = self._compute_updated_global_state([final_new_state["layer"]],
                                                                global_state)
      # for k in updated_params:
      #   print(k.shape)
      # print("====")
      # for k in updated_attention:
      #   print(k.shape)
      # print("====")
      # for s in updated_states:
      #   for k in s:
      #     print(k, s[k].shape)
      #   print("====")
      # print("====")
    return (updated_params, updated_states, [updated_global_state],
            updated_attention)

  '''
  def _compute_updates(self, params, grads, states, global_state):
    # Store the updated parameters and states.
    updated_params = []
    updated_attention = []
    updated_states = []

    with tf.variable_scope(opt.OPTIMIZER_SCOPE):

      mean_log_lr = self._compute_mean_log_lr(states)

      # Iterate over the layers.
      for param, grad_unflat, state in zip(params, grads, states):

        with tf.variable_scope("PerTensor", reuse=self.reuse_vars):
          self.reuse_vars = True
          grad = tf.reshape(grad_unflat, [-1, 1])

          # Create the RNN input. We will optionally extend it with additional
          # features such as curvature and gradient signal vs. noise.
          (grads_scaled, mean_squared_gradients,
           grads_accum) = self._compute_scaled_and_ms_grads(grad, state)
          rnn_input = [g for g in grads_scaled]

          self._extend_rnn_input(rnn_input, state, grads_scaled,
                                 mean_squared_gradients, mean_log_lr)

          # Concatenate any features we've collected.
          rnn_input_tensor = tf.concat(rnn_input, 1)

          layer_state, new_param_state = self._update_rnn_cells(
              state, global_state, rnn_input_tensor,
              len(rnn_input) != len(grads_scaled))

          (scl_decay, inp_decay, new_log_lr, update_step, lr_attend,
           attention_delta) = self._compute_rnn_state_projections(
               state, new_param_state, grads_scaled)

          # apply output constrain to bound the effective step size
          if self.use_output_constrain:
            update_step = self.output_constrain_alpha*tf.math.tanh(update_step)
          # Apply updates and store state variables.
          if self.use_attention:
            truth = state["true_param"]
            updated_param = truth - update_step
            attention_step = tf.reshape(lr_attend * attention_delta,
                                        truth.get_shape())
            updated_attention.append(truth - attention_step)
          else:
            updated_param = param - update_step
            updated_attention.append(updated_param)
          updated_params.append(updated_param)

          # Collect the new state.
          new_state = {
              "parameter": new_param_state,
              "scl_decay": scl_decay,
              "inp_decay": inp_decay,
              "true_param": updated_param,
          }
          if layer_state is not None:
            new_state["layer"] = layer_state

          if self.dynamic_output_scale:
            new_state["log_learning_rate"] = new_log_lr

          for i in range(self.num_gradient_scales):
            new_state["grad_accum{}".format(i + 1)] = grads_accum[i]
            new_state["ms{}".format(i + 1)] = mean_squared_gradients[i]
          updated_states.append(new_state)

      updated_global_state = self._compute_updated_global_state([layer_state],
                                                                global_state)

    return (updated_params, updated_states, [updated_global_state],
            updated_attention)
  '''

  def _compute_mean_log_lr(self, states):
    """Computes the mean log learning rate across all variables."""
    if self.use_problem_lr_mean and self.use_relative_lr:

      sum_log_lr = 0.
      count_log_lr = 0.
      for state in states:
        sum_log_lr += tf.reduce_sum(state["log_learning_rate"])
        # Note: get_shape().num_elements()=num elements in the original tensor.
        count_log_lr += state["log_learning_rate"].get_shape().num_elements()
      return sum_log_lr / count_log_lr

  def _compute_scaled_and_ms_grads(self, grad, state):
    """Computes the scaled gradient and the mean squared gradients.

    Gradients are also accumulated across different timescales if appropriate.

    Args:
      grad: The gradient tensor for this layer.
      state: The optimizer state for this layer.

    Returns:
      The scaled gradients, mean squared gradients, and accumulated gradients.
    """
    import pdb
    # pdb.set_trace()
    input_decays = [state["inp_decay"]]
    # print(input_decays)
    scale_decays = [state["scl_decay"]]
    # print(scale_decays)
    if self.use_multiple_scale_decays and self.num_gradient_scales > 1:
      for i in range(self.num_gradient_scales - 1):
        # print(tf.sqrt(scale_decays[i]))
        scale_decays.append(tf.sqrt(scale_decays[i]))

    for i in range(self.num_gradient_scales - 1):
      # print(tf.sqrt(input_decays[i]))
      # Each accumulator on twice the timescale of the one before.
      input_decays.append(tf.sqrt(input_decays[i]))
    grads_accum = []
    grads_scaled = []
    mean_squared_gradients = []

    # populate the scaled gradients and associated mean_squared values
    if self.num_gradient_scales > 0:
      for i, decay in enumerate(input_decays):
        if self.num_gradient_scales == 1:
          # We don't accumulate if no scales, just take the current gradient.
          grad_accum = grad
        else:
          # The state vars are 1-indexed.
          old_accum = state["grad_accum{}".format(i + 1)]
          grad_accum = grad * (1. - decay) + old_accum * decay

        grads_accum.append(grad_accum)

        sd = scale_decays[i if self.use_multiple_scale_decays else 0]
        grad_scaled, ms = utils.rms_scaling(grad_accum, sd,
                                            state["ms{}".format(i + 1)],
                                            update_ms=True)
        grads_scaled.append(grad_scaled)
        mean_squared_gradients.append(ms)

    return grads_scaled, mean_squared_gradients, grads_accum

  def _extend_rnn_input(self, rnn_input, state, grads_scaled,
                        mean_squared_gradients, mean_log_lr):
    """Computes additional rnn inputs and adds them to the rnn_input list."""
    if self.num_gradient_scales > 1 and self.use_grad_products:
      # This gives a measure of curvature relative to input averaging
      # lengthscale and to the learning rate
      grad_products = [a * b for a, b in
                       zip(grads_scaled[:-1], grads_scaled[1:])]
      rnn_input.extend([g for g in grad_products])

    if self.use_log_means_squared:
      log_means_squared = [tf.log(ms + 1e-16)
                           for ms in mean_squared_gradients]

      avg = tf.reduce_mean(log_means_squared, axis=0)
      # This gives a measure of the signal vs. noise contribution to the
      # gradient, at the current averaging lengthscale. If all the noise
      # is averaged out, and if updates are small, these will be 0.
      mean_log_means_squared = [m - avg for m in log_means_squared]

      rnn_input.extend([m for m in mean_log_means_squared])

    if self.use_relative_lr or self.use_extreme_indicator:
      if not self.dynamic_output_scale:
        raise Exception("Relative LR and Extreme Indicator features "
                        "require dynamic_output_scale to be set to True.")
      log_lr_vec = tf.reshape(state["log_learning_rate"], [-1, 1])
      if self.use_relative_lr:
        if self.use_problem_lr_mean:
          # Learning rate of this dimension vs. rest of target problem.
          relative_lr = log_lr_vec - mean_log_lr
        else:
          # Learning rate of this dimension vs. rest of tensor.
          relative_lr = log_lr_vec - tf.reduce_mean(log_lr_vec)
        rnn_input.append(relative_lr)
      if self.use_extreme_indicator:
        # Indicator of extremely large or extremely small learning rate.
        extreme_indicator = (tf.nn.relu(log_lr_vec - tf.log(1.)) -
                             tf.nn.relu(tf.log(1e-6) - log_lr_vec))
        rnn_input.append(extreme_indicator)

    if self.use_lr_shortcut:
      log_lr_vec = tf.reshape(state["log_learning_rate"], [-1, 1])
      rnn_input.append(log_lr_vec - tf.log(1e-3))

  def _update_rnn_cells(self, state, global_state, rnn_input_tensor,
                        use_additional_features):
    """Updates the component RNN cells with the given state and tensor.

    Args:
      state: The current state of the optimizer.
      global_state: The current global RNN state.
      rnn_input_tensor: The input tensor to the RNN.
      use_additional_features: Whether the rnn input tensor contains additional
          features beyond the scaled gradients (affects whether the rnn input
          tensor is used as input to the RNN.)

    Returns:
      layer_state: The new state of the per-tensor RNN.
      new_param_state: The new state of the per-parameter RNN.
    """
    # lowest level (per parameter)
    #   input -> gradient for this parameter
    #   bias -> output from the layer RNN
    with tf.variable_scope("Layer0_RNN"):
      total_bias = None
      if self.num_layers > 1:
        sz = 3 * self.cells[0].state_size    # size of the concatenated bias
        param_bias = utils.affine([state["layer"]], sz,
                                  scope="Param/Affine",
                                  scale=FLAGS.hrnn_affine_scale,
                                  random_seed=self.random_seed)
        total_bias = param_bias
        if self.num_layers == 3:
          global_bias = utils.affine(global_state, sz,
                                     scope="Global/Affine",
                                     scale=FLAGS.hrnn_affine_scale,
                                     random_seed=self.random_seed)
          total_bias += global_bias

      new_param_state, _ = self.cells[0](
          rnn_input_tensor, state["parameter"], bias=total_bias)

    if self.num_layers > 1:
      # middle level (per layer)
      #   input -> average hidden state from each parameter in this layer
      #   bias -> output from the RNN at the global level
      with tf.variable_scope("Layer1_RNN"):
        if not use_additional_features:
          # Restore old behavior and only add the mean of the new params.
          layer_input = tf.reduce_mean(new_param_state, 0, keep_dims=True)
        else:
          layer_input = tf.reduce_mean(
              tf.concat((new_param_state, rnn_input_tensor), 1), 0,
              keep_dims=True)
        if self.num_layers == 3:
          sz = 3 * self.cells[1].state_size
          layer_bias = utils.affine(global_state, sz,
                                    scale=FLAGS.hrnn_affine_scale,
                                    random_seed=self.random_seed)
          layer_state, _ = self.cells[1](
              layer_input, state["layer"], bias=layer_bias)
        else:
          layer_state, _ = self.cells[1](layer_input, state["layer"])
    else:
      layer_state = None

    return layer_state, new_param_state

  def _compute_rnn_state_projections(self, state, new_param_state,
                                     grads_scaled):
    """Computes the RNN state-based updates to parameters and update steps."""
    # Compute the update direction (a linear projection of the RNN output).
    update_weights = self.update_weights

    update_delta = utils.project(new_param_state, update_weights)
    if self.use_gradient_shortcut:
      # Include an affine projection of just the direction of the gradient
      # so that RNN hidden states are freed up to store more complex
      # functions of the gradient and other parameters.
      grads_scaled_tensor = tf.concat([g for g in grads_scaled], 1)
      update_delta += utils.affine(grads_scaled_tensor, 1,
                                   scope="GradsToDelta",
                                   include_bias=False,
                                   vec_mean=1. / len(grads_scaled),
                                   random_seed=self.random_seed)
    if self.dynamic_output_scale:
      denom = tf.sqrt(tf.reduce_mean(update_delta ** 2) + 1e-16)

      update_delta /= denom

    if self.use_attention:
      attention_weights = self.attention_weights
      attention_delta = utils.project(new_param_state,
                                      attention_weights)
      if self.use_gradient_shortcut:
        attention_delta += utils.affine(grads_scaled_tensor, 1,
                                        scope="GradsToAttnDelta",
                                        include_bias=False,
                                        vec_mean=1. / len(grads_scaled),
                                        random_seed=self.random_seed)
      if self.dynamic_output_scale:
        attention_delta /= tf.sqrt(
            tf.reduce_mean(attention_delta ** 2) + 1e-16)
    else:
      attention_delta = None

    # The updated decay is an affine projection of the hidden state.
    scl_decay = utils.project(new_param_state, self.scl_decay_weights,
                              bias=self.scl_decay_bias,
                              activation=tf.nn.sigmoid)
    # This is only used if learnable_decay and num_gradient_scales > 1
    inp_decay = utils.project(new_param_state, self.inp_decay_weights,
                              bias=self.inp_decay_bias,
                              activation=tf.nn.sigmoid)

    # Also update the learning rate.
    lr_param, lr_attend, new_log_lr = self._compute_new_learning_rate(
        state, new_param_state)

    # update_step = tf.reshape(lr_param * update_delta,
    #                          state["true_param"].get_shape())

    update_step = lr_param * update_delta

    return (scl_decay, inp_decay, new_log_lr, update_step, lr_attend,
            attention_delta)

  def _compute_new_learning_rate(self, state, new_param_state):
    if self.dynamic_output_scale:
      # Compute the change in learning rate (an affine projection of the
      # RNN state, passed through a sigmoid or log depending on flags).
      # Update the learning rate, w/ momentum.
      lr_change = utils.project(new_param_state, self.lr_weights,
                                bias=self.lr_bias)
      step_log_lr = state["log_learning_rate"] + lr_change

      # Clip the log learning rate to the flag at the top end, and to
      # (log(min int32) - 1) at the bottom

      # Check out this hack: we want to be able to compute the gradient
      # of the downstream result w.r.t lr weights and bias, even if the
      # value of step_log_lr is outside the clip range. So we clip,
      # subtract off step_log_lr, and wrap all that in a stop_gradient so
      # TF never tries to take the gradient of the clip... or the
      # subtraction. Then we add BACK step_log_lr so that downstream still
      # receives the clipped value. But the GRADIENT of step_log_lr will
      # be the gradient of the unclipped value, which we added back in
      # after stop_gradients.
      step_log_lr += tf.stop_gradient(
          tf.clip_by_value(step_log_lr, -33, self.max_log_lr)
          - step_log_lr)

      lr_momentum_logit = tf.get_variable(
          "learning_rate_momentum_logit",
          initializer=FLAGS.learning_rate_momentum_logit_init)
      lrm = tf.nn.sigmoid(lr_momentum_logit)
      new_log_lr = (lrm * state["log_learning_rate"] +
                    (1. - lrm) * step_log_lr)
      param_stepsize_offset = tf.get_variable("param_stepsize_offset",
                                              initializer=-1.)
      lr_param = tf.exp(step_log_lr + param_stepsize_offset)
      lr_attend = tf.exp(step_log_lr) if self.use_attention else lr_param
    else:
      # Dynamic output scale is off, LR param is always 1.
      lr_param = 2. * utils.project(new_param_state, self.lr_weights,
                                    bias=self.lr_bias,
                                    activation=tf.nn.sigmoid)
      new_log_lr = None
      lr_attend = lr_param

    return lr_param, lr_attend, new_log_lr

  def _compute_updated_global_state(self, layer_states, global_state):
    """Computes the new global state gives the layers states and old state.

    Args:
      layer_states: The current layer states.
      global_state: The old global state.

    Returns:
      The updated global state.
    """
    updated_global_state = []
    if self.num_layers == 3:
      # highest (global) layer
      #   input -> average hidden state from each layer-specific RNN
      #   bias -> None
      with tf.variable_scope("Layer2_RNN", reuse=tf.AUTO_REUSE):
        self.reuse_global_state = True
        global_input = tf.reduce_mean(tf.concat(layer_states, 0), 0,
                                      keep_dims=True)
        updated_global_state, _ = self.cells[2](global_input, global_state[0])
    return updated_global_state

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Overwrites the tf.train.Optimizer interface for applying gradients."""

    # Pull out the variables.
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
    for g, v in grads_and_vars:
      if not isinstance(g, (tf.Tensor, tf.IndexedSlices, type(None))):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      if not isinstance(v, tf.Variable):
        raise TypeError(
            "Variable must be a tf.Variable: %s" % v)
      if g is not None:
        self._assert_valid_dtypes([g, v])
    var_list = [v for g, v in grads_and_vars if g is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s" %
                       (grads_and_vars,))

    # Create slots for the variables.
    with tf.control_dependencies(None):
      self._create_slots(var_list)

    # Store update ops in this list.
    with tf.op_scope([], name, self._name) as name:

      # Prepare the global state.
      with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
        gs = self._initialize_global_state()
        if gs:
          global_state = [tf.get_variable("global_state", initializer=gs[0])]
        else:
          global_state = []

      # Get the states for each variable in the list.
      states = [{key: self.get_slot(var, key) for key in self.get_slot_names()}
                for var in var_list]

      # Compute updated values.
      grads, params = zip(*grads_and_vars)
      args = (params, grads, states, global_state)
      updates = self._compute_updates(*args)
      new_params, new_states, new_global_state, new_attention = updates
      # Assign op for new global state.
      update_ops = [tf.assign(gs, ngs)
                    for gs, ngs in zip(global_state, new_global_state)]

      # Create the assign ops for the params and state variables.
      args = (params, states, new_params, new_attention, new_states)
      for var, state, new_var, new_var_attend, new_state in zip(*args):
        # Assign updates to the state variables.
        state_assign_ops = [tf.assign(state_var, new_state[key])
                            for key, state_var in state.items()]

        # Update the parameter.
        with tf.control_dependencies(state_assign_ops):
          if self.use_attention:
            # Assign to the attended location, rather than the actual location
            # so that the gradients are computed where attention is.
            param_update_op = var.assign(new_var_attend)
          else:
            param_update_op = var.assign(new_var)

        with tf.name_scope("update_" + var.op.name):   #, tf.colocate_with(var):
          update_ops.append(param_update_op)

      real_params = [self.get_slot(var, "true_param") for var in var_list]

      if global_step is None:
        # NOTE: if using the optimizer in a non-test-optimizer setting (e.g.
        # on Inception), remove the real_params return value. Otherwise
        # the code will throw an error.
        return self._finish(update_ops, name), real_params
      else:
        with tf.control_dependencies([self._finish(update_ops, "update")]):
          return state_ops.assign_add(global_step, 1, name=name).op, real_params
