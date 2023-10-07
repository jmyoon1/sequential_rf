# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any
import copy

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
from flax.metrics import tensorboard
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, iddpm
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import datetime
import wandb
import matplotlib.pyplot as plt

import jax_smi

FLAGS = flags.FLAGS


def train(config, workdir, log_name):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  # ====================================================================================================== #
  # Get logger
  jax_smi.initialise_tracking()

  # wandb_dir: Directory of wandb summaries
  current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  if log_name is None:
    wandb.init(project="sequential_rf", name=f"{config.model.name}-{current_time}", entity="seqrf", resume="allow")
  else:
    wandb.init(project="sequential_rf", name=log_name, entity="seqrf", resume="allow")
  wandb_dir = os.path.join(workdir, "wandb")
  tf.io.gfile.makedirs(wandb_dir)
  wandb.config = config

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  rng = jax.random.PRNGKey(config.seed)
  # ====================================================================================================== #
  # Get modes
  """
    train_mode
      'train_baseline':  Train baseline model (1-Rectified flow) before Reflow.
      'gen_reflow_data': Reflow data generating phase before k-RF (k > 1)
      'train_reflow':    Training with reflowed data.
  """
  if_reflow = False if 'rf_phase' not in config.model else (config.model.rf_phase > 1)
  if config.model.rf_phase == 1:
    train_mode = 'train_baseline'
  elif (config.model.rf_phase > 1) and (config.training.reflow_mode == 'gen_reflow'):
    train_mode = 'gen_reflow_data'
  elif (config.model.rf_phase > 1) and (config.training.reflow_mode == 'train_reflow'):
    train_mode = 'train_reflow'
    reflow_batch_idx = 0
    n_total_data = 0
  elif (config.model.rf_phase > 1) and (config.training.reflow_mode == 'train_distill'):
    train_mode = 'train_distill'
    reflow_batch_idx = 0
    n_total_data = 0
  else:
    raise NotImplementedError()
  # ====================================================================================================== #
  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  state = mutils.init_train_state(step_rng, config)
  if train_mode != 'train_baseline' and train_mode != 'train_distill':
    if config.model.rf_phase == 2: # k = 2: start from baseline model
      old_checkpoint_dir = os.path.join(workdir, "checkpoints")
    else: # k > 2: start from (k-1)-RF
      old_checkpoint_dir = os.path.join(workdir, f"checkpoints_reflow_{config.model.rf_phase - 1}")
    checkpoint_dir = os.path.join(workdir, f"checkpoints_reflow_{config.model.rf_phase}")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, f"checkpoints-meta_reflow_{config.model.rf_phase}")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(checkpoint_meta_dir)
    # # Resume training when intermediate checkpoints are detected
    state = checkpoints.restore_checkpoint(old_checkpoint_dir, state, step=config.training.reflow_source_ckpt)
    state = state.replace(step=0)

    # If exists, resume from reflow checkpoint when detected
    state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)

  elif train_mode == 'train_distill':
    assert config.model.rf_phase >= 3
    old_checkpoint_dir = os.path.join(workdir, f"checkpoints_reflow_{config.model.rf_phase - 1}")
    checkpoint_dir = os.path.join(workdir, f"checkpoints_distill_{config.model.rf_phase}")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, f"checkpoints-meta_distill_{config.model.rf_phase}")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(checkpoint_meta_dir)
    # # Resume training when intermediate checkpoints are detected
    state = checkpoints.restore_checkpoint(old_checkpoint_dir, state, step=config.training.reflow_source_ckpt)
    state = state.replace(step=0)

  else:
    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(checkpoint_meta_dir)
    # Resume training when intermediate checkpoints are detected
    state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  # ====================================================================================================== #
  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  # ====================================================================================================== #
  # Setup SDEs
  if config.training.sde.lower() == 'rfsde':
    # TODO: We only use this
    sde = sde_lib.RFSDE(N=config.model.num_scales)
    sampling_eps = 1e-3 # Not used.
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  # ====================================================================================================== #
  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(config, sde, state, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)

  if train_mode != 'gen_reflow_data':
    eval_step_fn = losses.get_step_fn(config, sde, state, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)
    # Pmap (and jit-compile) multiple evaluation steps together for faster running
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

  # Building sampling functions
  if (config.training.snapshot_sampling or config.training.snapshot_statistics) and (train_mode != 'gen_reflow_data'):
    sampling_shape = (config.eval.batch_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, state, sampling_shape, inverse_scaler, sampling_eps)
  
  elif train_mode == 'gen_reflow_data':
    sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, state, sampling_shape, inverse_scaler, sampling_eps, gen_reflow=True)

  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters
  # ====================================================================================================== #
  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.process_index() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.process_index())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"
  # ====================================================================================================== #
  # Trigger that update reflow batch
  n_reflow_batch = config.training.n_reflow_data // (config.training.batch_size * config.training.n_jitted_steps) + 1
  reflow_sample_dir = os.path.join(sample_dir, "reflow_batch")

  def gen_reflow_pair(rng, batch):
    """
      Input
        rng: jax.random.PRNGKey variable.
        batch: input batch, scaled by `scaler`.
    """
    average_nfe = 0.0
    nfe_count = 0.0

    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)

    """
      config.training.reflow_t: how many division to divide reflow_t
      For example, divide t into
        0 : randomly divide into (a, b)
        1 : [0, 1]
        2 : [0, 0.5], [0.5, 1]
        3 : [0, 1/3], [1/3, 2/3], [2/3, 1]
        4 : [0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]
      and generate data from latter time to former time, with equal probability

      Return:
        (x0_batch, x1_batch), (t0_batch, t1_batch)
        x0_batch: destination x at time t0, inverse_scaled to [0, 1]
        x1_batch: source x at time t1, inverse_scaled to [0, 1]
        t0_batch: destination time
        t1_batch: source time
    """
    for j in range(config.training.n_jitted_steps):
      (x0, x1), (t0, t1), _ = sampling_fn(next_rng, pstate, cond_image=batch[:, j])
      x0, x1 = jnp.reshape(x0, (-1, *x0.shape[-3:])), jnp.reshape(x1, (-1, *x1.shape[-3:]))
      t0, t1 = jnp.reshape(t0, (-1)), jnp.reshape(t1, (-1))
      if j == 0:
        x0_batch, x1_batch = x0, x1
        t0_batch, t1_batch = t0, t1
      else:
        x0_batch, x1_batch = np.concatenate([x0_batch, x0], axis=0), np.concatenate([x1_batch, x1], axis=0)
        t0_batch, t1_batch = np.concatenate([t0_batch, t0], axis=0), np.concatenate([t1_batch, t1], axis=0)

    x0_batch, x1_batch = jnp.reshape(x0_batch, (-1, *x0_batch.shape[-3:])), jnp.reshape(x1_batch, (-1, *x1_batch.shape[-3:]))
    t0_batch, t1_batch = jnp.reshape(t0_batch, (-1)), jnp.reshape(t1_batch, (-1))

    return (scaler(x0_batch), scaler(x1_batch)), (t0_batch, t1_batch)

  def set_reflow_batch_fn(rng, train_iter):
    tf.io.gfile.makedirs(reflow_sample_dir)
    _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(reflow_sample_dir, "reflow_batch_*.npz"))] # Reset
    logging.info(f"Start generating reflow pair of {n_reflow_batch * config.training.batch_size * config.training.n_jitted_steps} data points.")

    for i in range(1, n_reflow_batch + 1):
      rng, step_rng = jax.random.split(rng)
      batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))
      (x0_batch, x1_batch), (t0_batch, t1_batch) = gen_reflow_pair(rng=step_rng, batch=batch['image'])
      np.savez_compressed(os.path.join(reflow_sample_dir, f"reflow_batch_{i}.npz"),
                          x0=x0_batch,
                          x1=x1_batch,
                          t0=t0_batch,
                          t1=t1_batch)
      
      if i == 1:
        # Draw sample pair figure
        x0_batch_draw = inverse_scaler(jnp.reshape(x0_batch, (-1, *x0_batch.shape[-3:]))[0:64])
        x1_batch_draw = inverse_scaler(jnp.reshape(x1_batch, (-1, *x1_batch.shape[-3:]))[0:64])
        utils.draw_figure_grid(x0_batch_draw, reflow_sample_dir, f"reflow_destination_example")
        utils.draw_figure_grid(x1_batch_draw, reflow_sample_dir, f"reflow_source_example")
        original_batch = jnp.reshape(jnp.swapaxes(batch['image'], 0, 1), (-1, *batch['image'].shape[-3:]))
        utils.draw_figure_grid(inverse_scaler(original_batch)[0:64], reflow_sample_dir, f"reflow_original_example")
        
      del x0_batch, x1_batch, t0_batch, t1_batch
      logging.info(f"Generated reflow pair {i}.")
  # ====================================================================================================== #
  # Main training or generation part
  if train_mode == "gen_reflow_data":
    set_reflow_batch_fn(rng, train_iter)
  else:
    for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
      if train_mode == "train_reflow" or train_mode == "train_distill":
        # Use pre-trained reflow dataset
        if step == initial_step:
          logging.info(f"Already have reflow data with {config.training.reflow_t} divisions.")
          reflow_data_files = tf.io.gfile.glob(os.path.join(reflow_sample_dir, "reflow_batch_*.npz"))
          assert len(reflow_data_files) == n_reflow_batch, \
            f"Have {len(reflow_data_files)} files; Should have {n_reflow_batch} reflow batches."
          for reflow_data_file in reflow_data_files:
            rf_data_temp = np.load(reflow_data_file)
            n_total_data += rf_data_temp['x0'].shape[0]
          logging.info(f"Have {n_total_data} simulation-driven reflow data.")
        reflow_batch = np.load(os.path.join(reflow_sample_dir, f"reflow_batch_{reflow_batch_idx+1}.npz"))
        batch = (
          (reflow_batch['x0'], reflow_batch['x1']),
          (reflow_batch['t0'], reflow_batch['t1'])
        )
        assert batch[0][0].shape[0] == config.training.n_jitted_steps * config.training.batch_size, \
          f"{batch[0][0].shape[0]} vs. {config.training.n_jitted_steps * config.training.batch_size}"
        batch = (
          jax.tree_map(
          lambda x: jnp.reshape(x, (jax.local_device_count(), config.training.n_jitted_steps, config.training.batch_size // jax.local_device_count(), *x.shape[-3:])),
          batch[0]
          ),
          jax.tree_map(
          lambda x: jnp.reshape(x, (jax.local_device_count(), config.training.n_jitted_steps, config.training.batch_size // jax.local_device_count())),
          batch[1]
          )
        )
      elif train_mode == "train_baseline":
        # Use batch
        batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))
      else:
        raise ValueError("train_mode should be in [`train_baseline`, `train_reflow`.]")

      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, pstate), ploss = p_train_step((next_rng, pstate), batch)

      if if_reflow:
        reflow_batch_idx = (reflow_batch_idx + 1) % n_reflow_batch

      # Calculate loss and save
      loss = flax.jax_utils.unreplicate(ploss).mean()
      wandb_log_dict = {'train/loss': float(loss)}
      # Log to console, file and tensorboard on host 0
      if jax.process_index() == 0 and step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss))
        wandb.log(wandb_log_dict, step=step)

      # Save a temporary checkpoint to resume training after pre-emption periodically
      if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.process_index() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                    step=step // config.training.snapshot_freq_for_preemption,
                                    keep=1)
      # ====================================================================================================== #
      # Report the loss on an evaluation dataset periodically only in train_baseline case
      if (step % config.training.eval_freq == 0) and (train_mode == "train_baseline"):
        eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        
        # Eval loss at the baseline.
        (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)

        eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
        wandb_log_dict = {'eval/loss': float(eval_loss)}
        if jax.process_index() == 0:
          logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
          wandb.log(wandb_log_dict, step=step)
      # ====================================================================================================== #
      # Save a checkpoint periodically and generate samples if needed
      if step % config.training.snapshot_freq == 0 or step == num_train_steps:
        # Save the checkpoint.
        if step != 0 and step % config.training.snapshot_save_freq == 0:
          if jax.process_index() == 0:
            saved_state = flax_utils.unreplicate(pstate)
            checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                        step=step // config.training.snapshot_save_freq,
                                        keep=np.inf)
        
        # ====================================================================================================== #
        # Generate and save one batch of samples
        if config.training.snapshot_sampling:
          rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
          sample_rng = jnp.asarray(sample_rng)
          (sample, init_noise), _, _ = sampling_fn(sample_rng, pstate)
          image_grid = jnp.reshape(sample, (-1, *sample.shape[-3:]))

          # Draw snapshot figure
          if train_mode == 'train_reflow':
            this_sample_dir = os.path.join(
              sample_dir, "reflow_{}_iter_{}_host_{}".format(config.model.rf_phase, step, jax.process_index()))
          elif train_mode == 'train_baseline':
            this_sample_dir = os.path.join(
              sample_dir, "iter_{}_host_{}".format(step, jax.process_index()))
          elif train_mode == 'train_distill':
            this_sample_dir = os.path.join(
              sample_dir, "distill_{}_iter_{}_host_{}".format(config.model.rf_phase, step, jax.process_index()))
          else:
            raise ValueError()
          tf.io.gfile.makedirs(this_sample_dir)
          utils.draw_figure_grid(image_grid, this_sample_dir, f"sample_{step}")
        # ====================================================================================================== #
        # Get statistics
        if config.training.snapshot_statistics:
          stats = utils.get_samples_and_statistics(config, rng, sampling_fn, pstate, sample_dir, sampling_shape, mode='train')
          logging.info(f"FID = {stats['fid']}")
          logging.info(f"KID = {stats['kid']}")
          logging.info(f"Inception_score = {stats['is']}")
          wandb_statistics_dict = {
            'fid': float(stats['fid']),
            'kid': float(stats['kid']),
            'inception_score': float(stats['is']),
          }
          wandb.log(wandb_statistics_dict, step=step)

          if train_mode == 'train_reflow':
            logging.info(f"straightness = {stats['straightness']['straightness']}")
            logging.info(f"sequential straightness = {stats['straightness']['seq_straightness']}")

            # Draw figures on absolute and relative straightness, and save.
            straightness_dir = os.path.join(sample_dir, "straightness")
            tf.io.gfile.makedirs(straightness_dir)
            t = jnp.linspace(sde.T, 0.0, sde.N)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set(xlim = [0, sde.T], title=f'straightness, step {step}', xlabel='t', ylabel='straightness')
            ax.plot(t, stats['straightness']['straightness_by_t'])
            plt.savefig(os.path.join(straightness_dir, f'straightness_{step}.png'))
            plt.close(fig)

            wandb_statistics_dict = {
              'straightness': float(stats['straightness']['straightness']),
              'seq_straightness': float(stats['straightness']['seq_straightness'])
            }
            wandb.log(wandb_statistics_dict, step=step)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set(xlim = [0, sde.T], title=f'sequential straightness, step {step}', xlabel='t', ylabel='straightness')
            ax.plot(t, stats['straightness']['seq_straightness_by_t'])
            plt.savefig(os.path.join(straightness_dir, f'straightness_{step}_sequential.png'))
            plt.close(fig)
        # ====================================================================================================== #

def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # ====================================================================================================== #
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  rng = jax.random.PRNGKey(config.seed + 1)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=1,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  # ====================================================================================================== #
  # Get modes
  """
    train_mode
      'train_baseline':  Train baseline model (1-Rectified flow) before Reflow.
      'gen_reflow_data': Reflow data generating phase before k-RF (k > 1)
      'train_reflow':    Training with reflowed data.
  """
  if_reflow = False if 'rf_phase' not in config.model else (config.model.rf_phase > 1)
  if config.model.rf_phase == 1:
    train_mode = 'train_baseline'
  elif (config.model.rf_phase > 1) and (config.training.reflow_mode == 'gen_reflow'):
    train_mode = 'gen_reflow_data'
  elif (config.model.rf_phase > 1) and (config.training.reflow_mode == 'train_reflow'):
    train_mode = 'train_reflow'
    reflow_batch_idx = 0
    n_total_data = 0
  elif (config.model.rf_phase > 1) and (config.training.reflow_mode == 'train_distill'):
    train_mode = 'train_distill'
    reflow_batch_idx = 0
    n_total_data = 0
  else:
    raise NotImplementedError()
  # ====================================================================================================== #
  # Initialize model
  rng, step_rng = jax.random.split(rng)
  state = mutils.init_train_state(step_rng, config)

  if config.sampling.predictor == 'rf_solver' and train_mode == 'train_reflow':
    assert config.model.rf_phase > 1
    checkpoint_dir = os.path.join(workdir, f"checkpoints_reflow_{config.model.rf_phase}")
  elif config.sampling.predictor == 'rf_solver' and train_mode == 'train_distill':
    assert config.model.rf_phase > 2
    checkpoint_dir = os.path.join(workdir, f"checkpoints_distill_{config.model.rf_phase}")
  else:
    checkpoint_dir = os.path.join(workdir, "checkpoints")
  # ====================================================================================================== #
  if config.training.sde.lower() == 'rfsde':
    sde = sde_lib.RFSDE(N=config.model.num_scales)
    sampling_eps = 1e-3 # Not used.
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  # ====================================================================================================== #
  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                      config.data.image_size, config.data.image_size,
                      config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, state, sampling_shape, inverse_scaler, sampling_eps)
  num_sampling_rounds = (config.eval.num_samples - 1) // config.eval.batch_size + 1
  # ====================================================================================================== #
  # Add additional task for evaluation (for example, get gradient statistics) here.
  # ====================================================================================================== #
  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.process_index())

  logging.info("begin checkpoint: %d" % (config.eval.begin_ckpt,))
  for ckpt in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
    logging.info(ckpt_filename)
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed and jax.process_index() == 0:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    try:
      state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
    except:
      time.sleep(60)
      try:
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
      except:
        time.sleep(120)
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
    # Replicate the training state for executing on multiple devices
    pstate = flax.jax_utils.replicate(state)
    # ====================================================================================================== #
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      state = jax.device_put(state)
      # Run sample generation for multiple rounds to create enough samples
      # Designed to be pre-emption safe. Automatically resumes when interrupted
      if jax.process_index() == 0:
        logging.info("Sampling -- checkpoint: %d" % (ckpt,))
      this_sample_dir = os.path.join(
        eval_dir, f"ckpt_{ckpt}_host_{jax.process_index()}")
      stats = utils.get_samples_and_statistics(config, rng, sampling_fn, pstate, this_sample_dir, sampling_shape, mode='eval')
      logging.info(f"FID = {stats['fid']}")
      logging.info(f"KID = {stats['kid']}")
      logging.info(f"Inception_score = {stats['is']}")
      if config.model.rf_phase > 1:
        logging.info(f"straightness = {stats['straightness']['straightness']}")

        # Draw figures on straightness and sequential straightness, and save.
        t = jnp.linspace(sde.T, sampling_eps, sde.N)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim = [0, sde.T], title=f'straightness', xlabel='t', ylabel='straightness')
        ax.plot(t, stats['straightness']['straightness_by_t'])
        plt.savefig(os.path.join(this_sample_dir, f'straightness.png'))
        plt.close(fig)
    # ====================================================================================================== #
