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

"""All functions related to loss computation and optimization.
"""

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from sde_lib import VESDE, VPSDE, I2SBSDE, RFSDE, EDMSDE
from utils import batch_mul
import optax
from models.layers import haar_downsample, haar_upsample


def get_optimizer(config):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    warmup = config.optim.warmup
    if warmup > 0:
      warmup_schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,
                                                           peak_value=config.optim.lr,
                                                           warmup_steps=warmup,
                                                           decay_steps=config.training.n_iters,
                                                           end_value=config.optim.lr)
    optimizer = optax.adamw(learning_rate=warmup_schedule, b1=config.optim.beta1, eps=config.optim.eps, weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(state,
                  grad,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    # Fast parameter update
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    state = state.apply_gradients(grads=clipped_grad) # variables['params'] updated.

    # Delayed parameter (EMA) update
    if config.model.variable_ema_rate:
      ema_decay = jnp.minimum(config.model.ema_rate, (1 + state.step) / (10 + state.step))
    else:
      ema_decay = config.model.ema_rate
    updates, new_opt_state_ema = state.tx_ema.update(
      state.params, state.opt_state_ema, ema_decay
    )
    state = state.replace(opt_state_ema=new_opt_state_ema)

    return state

  return optimize_fn


def get_edm_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, diftype=None, **kwargs):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
    type: None, 'wavelet', or 'conditional'

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """
    score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True, diftype=diftype)

    data = batch['image']

    rng, step_rng = random.split(rng)
    t = jnp.exp(sde.p_std * random.normal(step_rng, (data.shape[0],)) + sde.p_mean)
    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

    weight = (t ** 2 + sde.sigma_data ** 2) / (t * sde.sigma_data) ** 2
    losses = batch_mul(weight, jnp.square(score - data))

    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_rf_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, diftype=None, **kwargs):
  """
    Create a loss function for training RFs. (rectified flow)
    
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  """
  assert 'reflow' in kwargs

  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)

    """
      data configuration
      Output
        x = (original data, degraded data or noise)
    """
    if kwargs['reflow'] or kwargs['distill']:
      data = batch[0]
      t_ref = batch[1]
      x = data

      if kwargs['reflow']:
        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (x[0].shape[0],), minval=0.0, maxval=1) # interpolation level
      elif kwargs['distill']:
        t = jnp.ones((x[0].shape[0],))
      else:
        raise NotImplementedError()
      perturbed_data = batch_mul(x[0], 1 - t) + batch_mul(x[1], t)
      rng, step_rng = random.split(rng)
      score, new_model_state = score_fn(perturbed_data, t * t_ref[1] + (1 - t) * t_ref[0], step_rng)     

      # loss function
      score_ref = batch_mul(x[1] - x[0], 1 / (t_ref[1] - t_ref[0]))
      losses = jnp.square(score_ref - score)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

    else:
      data = batch['image']

      rng, step_rng = random.split(rng)
      z = random.normal(step_rng, data.shape)
      x = (data, z)
      
      rng, step_rng = random.split(rng)
      t = random.uniform(step_rng, (x[0].shape[0],), minval=0.0, maxval=sde.T)
      perturbed_data = batch_mul(x[0], 1 - t) + batch_mul(x[1], t)
      rng, step_rng = random.split(rng)
      score, new_model_state = score_fn(perturbed_data, t, step_rng)

      # loss function
      losses = jnp.square((x[1] - x[0]) - score)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_sb_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, diftype=None, **kwargs):
  """
    Create a loss function for training SBs. (Image-to-image Schrodinger bridge)
    
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
    diftype: None, 'wavelet', or 'conditional'
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
  if diftype == 'wavelet':
    sde_vp, sde_i2sb = sde['high'], sde['low']
    latency = kwargs['latency'] if 'latency' in kwargs.keys() else eps

  def loss_fn(rng, params, states, batch):
    data = batch['image']
    # SB and VP score functions
    if diftype == 'wavelet':
      model_fn = mutils.get_model_fn(model, params, states, train=train) # TODO: fix it. 
    elif diftype is None:
      score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True, diftype=diftype)
    else:
      raise NotImplementedError()

    if diftype == 'wavelet':
      # Perturb data, same as VP-type perturbation.
      # 1 --> 0.5: 128 sampling, 0.5 --> 0: 128 --> 256 sampling (with i2sb)
      data_low, data_high = haar_downsample(data)
      rng, step_rng = random.split(rng)
      t = random.uniform(step_rng, (data_high.shape[0],), minval=eps, maxval=sde_vp.T)
      rng, step_rng = random.split(rng)
      z_low = random.normal(step_rng, data_low.shape) # low-resolution random variable
      rng, step_rng = random.split(rng)
      z_high = random.normal(step_rng, data_high.shape) # high-resolution random variable
      
      assert latency >= eps
      # t_low: [eps, 0.5] --> [eps, latency], [0.5, 1] --> [latency, 1], low-resolution time of VP-type SDE
      # t_high: [eps, 0.5] --> [eps, 1], [0.5, 1] --> [1, 1], high-resolution time of VP-type SDE
      t_low = jnp.maximum((latency - eps) / (sde_vp.T / 2 - eps) * (t - sde_vp.T / 2) + latency, (sde_vp.T - latency) / (sde_vp.T / 2) * (t - sde_vp.T) + sde_vp.T)
      t_high = jnp.minimum((sde_vp.T - eps) / (sde_vp.T / 2 - eps) * (t - sde_vp.T / 2) + sde_vp.T, sde_vp.T * jnp.ones_like(t))
      t_mask_sb = jnp.heaviside(sde_vp.T / 2 - t, 0.5) # 1 at t = [0, sde.T / 2], 0 at t = [sde.T / 2, sde.T]
      t_mask_vp = jnp.heaviside(t - sde_vp.T / 2, 0.5) # 0 at t = [0, sde.T / 2], 1 at t = [sde.T / 2, sde.T]

      # t > 0.5 (VP-type perturbation)
      mean_low_vp, std_low_vp = sde_vp.marginal_prob(data_low, t_low)
      perturbed_data_low_vp = mean_low_vp + batch_mul(std_low_vp, z_low)
      mean_high_vp, std_high_vp = sde_vp.marginal_prob(data_high, t_high)
      perturbed_data_high_vp = mean_high_vp + batch_mul(std_high_vp, z_high)

      # t < 0.5 (SB-type perturbation)
      t_low_half = jnp.ones_like(t) * latency
      t_high_half = jnp.ones_like(t) * sde_vp.T
      mean_low_half, std_low_half = sde_vp.marginal_prob(data_low, t_low_half)
      perturbed_data_low_half = mean_low_half + batch_mul(std_low_half, z_low)
      mean_high_half, std_high_half = sde_vp.marginal_prob(data_high, t_high_half)
      perturbed_data_high_half = mean_high_half + batch_mul(std_high_half, z_high)
      perturbed_data_half = haar_upsample(perturbed_data_low_half, perturbed_data_high_half) # perturbed (degraded) data at time t = 0.5

      data_sb = (data, perturbed_data_half)
      mean_sb, std_sb = sde_i2sb.marginal_prob(data_sb, t)
      rng, step_rng = random.split(rng)
      z_sb = random.normal(step_rng, data.shape)

      # perturbed data and model function
      perturbed_data_vp = haar_upsample(perturbed_data_low_vp, perturbed_data_high_vp)
      perturbed_data_sb = mean_sb + batch_mul(std_sb, z_sb)
      perturbed_data = batch_mul(perturbed_data_vp, t_mask_vp) + batch_mul(perturbed_data_sb, t_mask_sb)
      t_divide = t + t_mask_vp * (sde_vp.T / 2)
      rng, step_rng = random.split(rng)
      _model, new_model_state = model_fn(perturbed_data, t_divide * 999, step_rng)

      # model function to score function
      model_low_vp, model_high_vp = haar_downsample(_model)
      losses_vp = jnp.concatenate([jnp.square(model_low_vp + z_low), jnp.square(model_high_vp + z_high)], axis=-1)
      losses_vp = batch_mul(losses_vp, t_mask_vp)
      losses_vp = reduce_op(losses_vp.reshape((losses_vp.shape[0], -1)), axis=-1)

      noise_sb = (perturbed_data - data) / sde_i2sb.get_sigma(t)
      losses_sb = jnp.square(_model - noise_sb)
      losses_sb = batch_mul(losses_sb, t_mask_sb)
      losses_sb = reduce_op(losses_sb.reshape((losses_sb.shape[0], -1)), axis=-1)

      losses = losses_vp + losses_sb
    
    elif diftype is None: # raw i2sb task
      if 'i2sb_task' in kwargs:
        if kwargs['i2sb_task'] == 'sr':
          assert 'from_res' in kwargs, "from_res should be defined."
          assert 'to_res' in kwargs, "to_res should be defined."
          from_res = kwargs['from_res']
          to_res = kwargs['to_res']

          high_sigma = kwargs['high_sigma'] if 'high_sigma' in kwargs else 0.0

          B, H, W, C = data.shape
          x0 = jax.image.resize(data, (B, from_res, from_res, C), method='nearest')
          if high_sigma > 0.0:
            rng, step_rng = random.split(rng)
            x0 += random.normal(step_rng, x0.shape) * high_sigma # noising degraded image
          x_degraded = jax.image.resize(x0, (B, to_res, to_res, C), method='nearest') # degraded image
          x_original = jax.image.resize(data, (B, to_res, to_res, C), method='nearest') # original image
          x = (x_original, x_degraded)

          # perturbed data and score function
          rng, step_rng = random.split(rng)
          t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
          mean, std = sde.marginal_prob(x, t)
          rng, step_rng = random.split(rng)
          z = random.normal(step_rng, data.shape)
          perturbed_data = mean + batch_mul(std, z)
          rng, step_rng = random.split(rng)
          score, new_model_state = score_fn(perturbed_data, t, step_rng)

          # loss function: Correspond to (12) in I2SB paper.
          score_hat = batch_mul(perturbed_data - x_original, 1. / sde.get_sigma(t))
          losses = jnp.square(score - score_hat)
          losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)

        else:
          raise NotImplementedError()
      else:
        raise ValueError("No specific i2sb task is denoted.")

    else:
      raise NotImplementedError()

    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, diftype=None, **kwargs):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
    type: None, 'wavelet', or 'conditional'

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """
    if diftype == 'wavelet':
      assert 'latency' in kwargs.keys()
      latency = kwargs['latency']
      score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True, diftype=diftype, latency=latency, eps=eps)
    else:
      score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True, diftype=diftype)

    data = batch['image']

    if diftype is None:
      rng, step_rng = random.split(rng)
      t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
      rng, step_rng = random.split(rng)
      z = random.normal(step_rng, data.shape)
      mean, std = sde.marginal_prob(data, t)
      perturbed_data = mean + batch_mul(std, z)
      rng, step_rng = random.split(rng)
      score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

      if not likelihood_weighting:
        losses = jnp.square(batch_mul(score, std) + z)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
      else:
        g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
        losses = jnp.square(score + batch_mul(z, 1. / std))
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    elif diftype == 'conditional':
      # 128 to 256, conditional.
      data_low, data_high = haar_downsample(data)
      rng, step_rng = random.split(rng)
      t = random.uniform(step_rng, (data_high.shape[0],), minval=eps, maxval=sde.T)
      rng, step_rng = random.split(rng)
      z = random.normal(step_rng, data_high.shape)
      mean, std = sde.marginal_prob(data_high, t)
      perturbed_data_high = mean + batch_mul(std, z)
      perturbed_data = haar_upsample(data_low, perturbed_data_high)
      rng, step_rng = random.split(rng)
      score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

      if not likelihood_weighting:
        losses = jnp.square(batch_mul(score, std) + z)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
      else:
        g2 = sde.sde(jnp.zeros_like(data_high), t)[1] ** 2
        losses = jnp.square(score + batch_mul(z, 1. / std))
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    elif diftype == 'wavelet':
      # 1 --> 0.5: 128 sampling, 0.5 --> 0: 128 --> 256 sampling
      data_low, data_high = haar_downsample(data)
      rng, step_rng = random.split(rng)
      t = random.uniform(step_rng, (data_high.shape[0],), minval=eps, maxval=sde.T)
      rng, step_rng = random.split(rng)
      z_low = random.normal(step_rng, data_low.shape)
      rng, step_rng = random.split(rng)
      z_high = random.normal(step_rng, data_high.shape)
      
      assert latency >= eps
      t_low = jnp.maximum((latency - eps) / (sde.T / 2 - eps) * (t - sde.T / 2) + latency, (sde.T - latency) / (sde.T / 2) * (t - sde.T) + sde.T)
      t_high = jnp.minimum((sde.T - eps) / (sde.T / 2 - eps) * (t - sde.T / 2) + sde.T, sde.T * jnp.ones_like(t))

      mean_low, std_low = sde.marginal_prob(data_low, t_low)
      perturbed_data_low = mean_low + batch_mul(std_low, z_low)
      mean_high, std_high = sde.marginal_prob(data_high, t_high)
      perturbed_data_high = mean_high + batch_mul(std_high, z_high)
      perturbed_data = haar_upsample(perturbed_data_low, perturbed_data_high)
      rng, step_rng = random.split(rng)
      score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)
      
      score_low, score_high = haar_downsample(score)
      if not likelihood_weighting:
        losses = jnp.concatenate([jnp.square(batch_mul(score_low, std_low) + z_low), jnp.square(batch_mul(score_high, std_high) + z_high)], axis=-1)
        losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
      else:
        g2_low = sde.sde(jnp.zeros_like(data_low), t_low)[1] ** 2
        losses_low = jnp.square(score_low + batch_mul(z_low, 1. / std_low))
        losses_low = reduce_op(losses_low.reshape((losses_low.shape[0], -1)), axis=-1) * g2_low
        g2_high = sde.sde(jnp.zeros_like(data_high), t_high)[1] ** 2
        losses_high = jnp.square(score_high + batch_mul(z_high, 1. / std_high))
        losses_high = reduce_op(losses_high.reshape((losses_high.shape[0], -1)), axis=-1) * g2_high
        losses = losses_low + losses_high

    else:
      raise NotImplementedError()



    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_smld_loss_fn(vesde, model, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = vesde.discrete_sigmas[::-1]
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    model_fn = mutils.get_model_fn(model, params, states, train=train)
    data = batch['image']
    rng, step_rng = random.split(rng)
    labels = random.choice(step_rng, vesde.N, shape=(data.shape[0],))
    sigmas = smld_sigma_array[labels]
    rng, step_rng = random.split(rng)
    noise = batch_mul(random.normal(step_rng, data.shape), sigmas)
    perturbed_data = noise + data
    rng, step_rng = random.split(rng)
    score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
    target = -batch_mul(noise, 1. / (sigmas ** 2))
    losses = jnp.square(score - target)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * sigmas ** 2
    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_ddpm_loss_fn(vpsde, model, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    model_fn = mutils.get_model_fn(model, params, states, train=train)
    data = batch['image']
    rng, step_rng = random.split(rng)
    labels = random.choice(step_rng, vpsde.N, shape=(data.shape[0],))
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)
    perturbed_data = batch_mul(sqrt_alphas_cumprod[labels], data) + \
                     batch_mul(sqrt_1m_alphas_cumprod[labels], noise)
    rng, step_rng = random.split(rng)
    score, new_model_state = model_fn(perturbed_data, labels, rng=step_rng)
    losses = jnp.square(score - noise)
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_step_fn(config, sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, **kwargs):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if isinstance(sde, EDMSDE):
    loss_fn = get_edm_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)

  elif isinstance(sde, I2SBSDE):
    loss_fn = get_sb_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                             continuous=True, likelihood_weighting=likelihood_weighting,
                             i2sb_task='sr', from_res=config.model.from_res, to_res=config.model.to_res,
                             high_sigma=config.model.high_sigma)

  elif isinstance(sde, RFSDE):
    loss_fn = get_rf_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                             continuous=True, likelihood_weighting=likelihood_weighting,
                             reflow=(config.training.reflow_mode=='train_reflow' and config.model.rf_phase > 1),
                             distill=(config.training.reflow_mode=='train_distill' and config.model.rf_phase > 1))

  elif continuous:
    if config.data.conditional:
      diftype = "conditional"
      loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                                continuous=True, likelihood_weighting=likelihood_weighting, diftype=diftype)
    else:
      diftype = None
      loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,
                                continuous=True, likelihood_weighting=likelihood_weighting, diftype=diftype)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.params
      states = state.states
      (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch)
      grad = jax.lax.pmean(grad, axis_name='batch')
      state = optimize_fn(state, grad)
      new_state = state.replace(
        step=state.step + 1,
        states=new_model_state,
      )
    else:
      loss, _ = loss_fn(step_rng, state.opt_state_ema.ema, state.states, batch)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return step_fn
