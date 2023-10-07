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

"""All functions and modules related to model definition.
"""
from typing import Any, Callable, Optional

import flax
import functools
import jax.numpy as jnp
import sde_lib
import jax
import optax
import numpy as np
from models import wideresnet_noise_conditional
from flax.training import checkpoints, train_state
from utils import batch_mul
from losses import optimization_manager, get_optimizer
from flax.core.frozen_dict import FrozenDict, freeze
from models.layers import haar_downsample, haar_upsample

_MODELS = {}

def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = jnp.exp(
    jnp.linspace(
      jnp.log(config.model.sigma_max), jnp.log(config.model.sigma_min),
      config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def init_train_state(rng, config) -> train_state.TrainState:
  # Get model parameters and definitions
  model_name = config.model.name
  model_def = functools.partial(get_model(model_name), config=config)
  model = model_def()

  input_shape = (jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels)
  label_shape = input_shape[:1]
  fake_input = jnp.zeros(input_shape)
  fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
  params_rng, dropout_rng = jax.random.split(rng)
  
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input, fake_label)

  # Create optimizer
  optimizer = get_optimizer(config)
  # optimizer_ema = optax.ema(decay=config.model.ema_rate)
  optimizer_ema = variable_ema()
  init_model_state, initial_params = variables.pop('params')
  class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: FrozenDict[str, Any]
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    tx_ema: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    opt_state_ema: optax.OptState
    states: Any
    lr: float = 0.0
    ema_rate: float = 0.0

    def apply_gradients(self, *, grads, **kwargs):
      """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

      Note that internally this function calls `.tx.update()` followed by a call
      to `optax.apply_updates()` to update `params` and `opt_state`.

      Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

      Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
      """
      updates, new_opt_state = self.tx.update(
          grads, self.opt_state, self.params)
      new_params = optax.apply_updates(self.params, updates)
      return self.replace(
          params=new_params,
          opt_state=new_opt_state,
          **kwargs,
      )

    @classmethod
    def create(cls, *, apply_fn, params, tx, tx_ema, **kwargs):
      """Creates a new instance with `step=0` and initialized `opt_state`."""
      opt_state = tx.init(params)
      opt_state_ema = tx_ema.init(params)
      return cls(
          step=0,
          apply_fn=apply_fn,
          params=params,
          tx=tx,
          tx_ema=tx_ema,
          opt_state=opt_state,
          opt_state_ema=opt_state_ema,
          **kwargs,
      )
  return TrainState.create(
    apply_fn=model.apply,
    params=variables['params'], # main parameter
    tx=optimizer,
    tx_ema=optimizer_ema, # EMA state that includes delayed parameter
    states=init_model_state,
    lr=config.optim.lr
  )


def get_model_fn(state, params, states, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    variables = {'params': params, **states}
    if not train:
      return state.apply_fn(variables, x, labels, train=False, mutable=False), states
    else:
      rngs = {'dropout': rng}
      return state.apply_fn(variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs)
      # if states:
      #   return outputs
      # else:
      #   return outputs, states

  return model_fn


def get_score_fn(sde, state, params, states, train=False, continuous=False, return_state=False, diftype=None, **kwargs):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all other mutable parameters.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    return_state: If `True`, return the new mutable states alongside the model output.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(state, params, states, train=train)

  """
    Description
      continuous or subVPSDE
        labels = t * 999 (1e-3 ~ 1 --> 0.999 ~ 999)
        model, state <- model_fn(x, labels)
        score <- - model / std
  """

  if isinstance(sde, dict):
    assert "low" in sde.keys()

    if isinstance(sde["low"], sde_lib.I2SBSDE):
      """
        In our implementation,
        sde is type of dict --> {
          "high": sde_lib.VPSDE,
          "low": sde_lib.I2SBSDE
        }
      """
      def score_fn(x, t, rng=None):
        
        if diftype == 'wavelet':

          # only work for `wavelet` case.`
          assert 'latency' in kwargs.keys()
          assert 'eps' in kwargs.keys()
          latency = kwargs['latency']
          eps = kwargs['eps']

        else:
          raise NotImplementedError()


    elif isinstance(sde["low"], sde_lib.RFSDE):
      """
        In our implementation,
        sde is type of dict --> {
          "high": sde_lib.VPSDE,
          "low": sde_lib.RFSDE
        }
      """
      def score_fn(x, t, rng=None):
        
        # When the SDE follows VP-I2SB
        if diftype == 'wavelet':
          # only work for `wavelet` case.`
          assert 'latency' in kwargs.keys()
          assert 'eps' in kwargs.keys()
          latency = kwargs['latency']
          eps = kwargs['eps']

          # In our implementation,
          # t_low: [eps, 0.5] to [eps, latency] // [0.5, 1] to [latency, 1]
          # t_high: [eps, 0.5] to [eps, 1] // [0.5, 1] to [1, 1]
          t_low = jnp.maximum((latency - eps) / (sde.T / 2 - eps) * (t - sde.T / 2) + latency, (sde.T - latency) / (sde.T / 2) * (t - sde.T) + sde.T)
          t_high = jnp.minimum((sde.T - eps) / (sde.T / 2 - eps) * (t - sde.T / 2) + sde.T, sde.T * jnp.ones_like(t))
          labels = t * 999 # {[0.999, 999] in vp case}

          t_mask_high = jnp.heaviside(t - sde.T / 2, 0.5)
          t_mask_low = jnp.heaviside(sde.T / 2 - t, 0.5)

          labels_sb = t * 999 # {[0.999, 499.5] if sb, [999, 1498.5] if vp}
          model, state = model_fn(x, labels_sb, rng) # t-domain of model_fn will be [0.999, 499.5] U [999, 1498.5]
          model_low, model_high = haar_downsample(model)

          # t > 0.5: use vp score function.
          std_low = sde['high'].marginal_prob(jnp.zeros_like(model_low), t_low)[1]
          std_high = sde['high'].marginal_prob(jnp.zeros_like(model_high), t_high)[1]

          score_low, score_high = batch_mul(-model_low, 1. / std_low), batch_mul(-model_high, 1. / std_high)
          score_vp = haar_upsample(score_low, score_high)

          # t <= 0.5: use i2sb score function.
          score_sb = score
          """
            compute_label = (xt - x0) / sigma_t
            loss = mse_loss(pred, label)
          """
          score = batch_mul(score_sb, t_mask_low) + batch_mul(score_vp, t_mask_high)

        else:
          raise NotImplementedError()

        if return_state:
          return score, state
        else:
          return score

  elif isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, rng=None):
      # Scale neural network output by standard deviation and flip sign

      if continuous or isinstance(sde, sde_lib.subVPSDE):
        """
          For VP-trained models, t=0 corresponds to the lowest noise level
          The maximum value of time embedding is assumed to 999 for
          continuously-trained models.

          model_fn is trained to follow z (std noise) in our case.
        """
        if diftype == 'wavelet':
          assert 'latency' in kwargs.keys()
          assert 'eps' in kwargs.keys()
          latency = kwargs['latency']
          eps = kwargs['eps']

          # t_low: [eps, 0.5] to [eps, latency] // [0.5, 1] to [latency, 1]
          # t_high: [eps, 0.5] to [eps, 1] // [0.5, 1] to [1, 1]
          t_low = jnp.maximum((latency - eps) / (sde.T / 2 - eps) * (t - sde.T / 2) + latency, (sde.T - latency) / (sde.T / 2) * (t - sde.T) + sde.T)
          t_high = jnp.minimum((sde.T - eps) / (sde.T / 2 - eps) * (t - sde.T / 2) + sde.T, sde.T * jnp.ones_like(t))
          labels = t * 999
          
          model, state = model_fn(x, labels, rng)
          model_low, model_high = haar_downsample(model)

          std_low = sde.marginal_prob(jnp.zeros_like(model_low), t_low)[1]
          std_high = sde.marginal_prob(jnp.zeros_like(model_high), t_high)[1]

          score_low, score_high = batch_mul(-model_low, 1. / std_low), batch_mul(-model_high, 1. / std_high)
          score = haar_upsample(score_low, score_high)
        elif diftype == 'conditional':
          labels = t * 999
          model, state = model_fn(x, labels, rng)
          model_low, model_high = haar_downsample(model)
          std = sde.marginal_prob(jnp.zeros_like(model_high), t)[1]
          model = model_high
          score = batch_mul(-model, 1. / std)
        elif diftype is None:
          labels = t * 999
          model, state = model_fn(x, labels, rng)
          std = sde.marginal_prob(jnp.zeros_like(x), t)[1]
          score = batch_mul(-model, 1. / std)
        else:
          raise ValueError()
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        model, state = model_fn(x, labels, rng)
        std = sde.sqrt_1m_alphas_cumprod[labels.astype(jnp.int32)]
        score = batch_mul(-model, 1. / std)

      if return_state:
        return score, state
      else:
        return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, rng=None):
      if continuous:
        labels = sde.marginal_prob(jnp.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = jnp.round(labels).astype(jnp.int32)

      score, state = model_fn(x, labels, rng)
      if return_state:
        return score, state
      else:
        return score

  elif isinstance(sde, sde_lib.EDMSDE):
    def score_fn(x, t, rng=None):
      """
      Input
        x: input
        t: noise level

        (1) neural network parameterization
        d_theta = model_fn(c_in * x, log(t) / 4)
        
        (2) denoiser parameterization
        score_theta = c_skip * x + c_out * d_theta
      
      Return
        score_theta, state
      """
      c_skip = sde.sigma_data ** 2 / (t ** 2 + sde.sigma_data ** 2)       # skip scaler to output denoiser
      c_out = t * sde.sigma_data / jnp.sqrt(t ** 2 + sde.sigma_data ** 2) # denoiser scaler
      c_in = 1 / jnp.sqrt(sde.sigma_data ** 2 + t ** 2)                   # input scaler
      c_noise = jnp.log(t) / 4                                            # noise scaler to neural network

      x_scaled = batch_mul(c_in, x)
      model, state = model_fn(x_scaled, c_noise, rng)
      score = batch_mul(c_skip, x) + batch_mul(c_out, model)
      
      if return_state:
        return score, state
      else:
        return score

  elif isinstance(sde, sde_lib.I2SBSDE) or isinstance(sde, sde_lib.RFSDE):
    def score_fn(x, t, rng=None):
      labels = t * 999
      score, state = model_fn(x, labels, rng)
      if return_state:
        return score, state
      else:
        return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def create_classifier(prng_key, batch_size, ckpt_path):
  """Create a noise-conditional image classifier.

  Args:
    prng_key: A JAX random state.
    batch_size: The batch size of input data.
    ckpt_path: The path to stored checkpoints for this classifier.

  Returns:
    classifier: A `flax.linen.Module` object that represents the architecture of the classifier.
    classifier_params: A dictionary that contains trainable parameters of the classifier.
  """
  input_shape = (batch_size, 32, 32, 3)
  classifier = wideresnet_noise_conditional.WideResnet(
    blocks_per_group=4,
    channel_multiplier=10,
    num_outputs=10
  )
  initial_variables = classifier.init({'params': prng_key, 'dropout': jax.random.PRNGKey(0)},
                                      jnp.ones(input_shape, dtype=jnp.float32),
                                      jnp.ones((batch_size,), dtype=jnp.float32), train=False)
  model_state, init_params = initial_variables.pop('params')
  classifier_params = checkpoints.restore_checkpoint(ckpt_path, init_params)
  return classifier, classifier_params


def get_logit_fn(classifier, classifier_params):
  """ Create a logit function for the classifier. """

  def preprocess(data):
    image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
    image_std = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
    return (data - image_mean[None, ...]) / image_std[None, ...]

  def logit_fn(data, ve_noise_scale):
    """Give the logits of the classifier.

    Args:
      data: A JAX array of the input.
      ve_noise_scale: time conditioning variables in the form of VE SDEs.

    Returns:
      logits: The logits given by the noise-conditional classifier.
    """
    data = preprocess(data)
    logits = classifier.apply({'params': classifier_params}, data, ve_noise_scale, train=False, mutable=False)
    return logits

  return logit_fn


def get_classifier_grad_fn(logit_fn):
  """Create the gradient function for the classifier in use of class-conditional sampling. """

  def grad_fn(data, ve_noise_scale, labels):
    def prob_fn(data):
      logits = logit_fn(data, ve_noise_scale)
      prob = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels].sum()
      return prob

    return jax.grad(prob_fn)(data)

  return grad_fn

from optax._src import base as obase
from optax._src import utils as outils
from optax._src.transform import EmaState, update_moment, bias_correction

def variable_ema(
  debias: bool = True,
  accumulator_dtype: Optional[Any] = None
) -> obase.GradientTransformation:
  """
    ema with variable ema rate. ema rate is entered in the update_fn rather than being initialiized.
  """
  accumulator_dtype = outils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return EmaState(
        count=jnp.zeros([], jnp.int32),
        ema=jax.tree_util.tree_map(
            lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

  def update_fn(updates, state, decay, params=None):
    del params
    updates = new_ema = update_moment(updates, state.ema, decay, order=1)
    count_inc = outils.safe_int32_increment(state.count)
    if debias:
      updates = bias_correction(new_ema, decay, count_inc)
    state_ema = outils.cast_tree(new_ema, accumulator_dtype)
    return updates, EmaState(count=count_inc, ema=state_ema)

  return obase.GradientTransformation(init_fn, update_fn)