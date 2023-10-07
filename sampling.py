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
# pytype: skip-file
"""Various sampling methods."""
import functools

import jax
import jax.numpy as jnp
import jax.random as random
import abc
import flax

from models.utils import get_score_fn
from scipy import integrate
import sde_lib
from utils import batch_mul, batch_add, from_flattened_numpy, to_flattened_numpy, jprint

from models import utils as mutils
from models.layers import haar_downsample, haar_upsample

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, model, shape, inverse_scaler, eps, **kwargs):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """
  denoise = kwargs['denoise'] if 'denoise' in kwargs else config.sampling.noise_removal
  run_last_step = kwargs['run_last_step'] if 'run_last_step' in kwargs else True
  gen_reflow = kwargs['gen_reflow'] if 'gen_reflow' in kwargs else False

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  model=model,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=denoise,
                                  eps=eps)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 model=model,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=denoise,
                                 eps=eps,
                                 run_last_step=run_last_step,
                                 is_rf_sampler=(config.sampling.predictor=='rf_solver'),
                                 gen_reflow=gen_reflow,
                                 reflow_t=config.training.reflow_t if 'reflow_t' in config.training else 1,
                                 soft_bound=config.training.soft_division,
                                 timestep_style=config.sampling.timestep_style)
  elif sampler_name.lower() == 'two_stage':
    predictor_high = get_predictor(config.sampling.predictor_high.lower())
    predictor_low = get_predictor(config.sampling.predictor_low.lower())
    sampling_fn = get_two_stage_sampler(sde=sde,
                                        model=model,
                                        shape=shape,
                                        predictor_high=predictor_high,
                                        predictor_low=predictor_low,
                                        inverse_scaler=inverse_scaler,
                                        n_steps=config.sampling.n_steps_each,
                                        probability_flow=config.sampling.probability_flow, 
                                        continuous=config.training.continuous,
                                        denoise=denoise,
                                        latency=config.data.latency,
                                        eps=eps)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    if isinstance(sde, sde_lib.VPSDE):
      self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn
    self.probability_flow = probability_flow

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the predictor.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the corrector.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    dt = -1. / self.rsde.N
    z = random.normal(rng, x.shape)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + batch_mul(diffusion, jnp.sqrt(-dt) * z)
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    f, G = self.rsde.discretize(x, t)
    z = random.normal(rng, x.shape)
    x_mean = x - f
    x = x_mean + batch_mul(G, z)
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = jnp.where(timestep == 0, jnp.zeros(t.shape), sde.discrete_sigmas[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + batch_mul(score, sigma ** 2 - adjacent_sigma ** 2)
    std = jnp.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, noise)
    return x, x_mean

  def vpsde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    beta = sde.discrete_betas[timestep]
    score = self.score_fn(x, t)
    x_mean = batch_mul((x + batch_mul(beta, score)), 1. / jnp.sqrt(1. - beta))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(jnp.sqrt(beta), noise)
    return x, x_mean

  def update_fn(self, rng, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(rng, x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(rng, x, t)


@register_predictor(name='ddim')
class DDIMPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    sde = self.sde
    current_t, next_t = t
    z = self.score_fn(x, current_t)
    current_alpha, current_sigma, _ = sde.snr(current_t)
    next_alpha, next_sigma, _ = sde.snr(next_t)
    A_signal = next_alpha / current_alpha
    A_score = current_sigma ** 2 * A_signal - current_sigma * next_sigma
    x = batch_mul(A_signal, x) + batch_mul(A_score, z)
    return x, x


@register_predictor(name='second_order_heun')
class SecondOrderHeunPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    """
      Pseudocode
        1. Sample x_0 ~ N(0, sigma^2 (t_0) s^2 (t_0) I)
        2. for i in [0, N-1]:
             d_i <- (sigma' / sigma + s' / s)(t) * x_i - (sigma' * s / sigma)(t) D_theta(x_i / s(t) ; sigma(t)), t=t_i       # Evaluate dx/dt at t_i
             x_{i+1} <- x_i + (t_{i+1} - t_i) * d_i                                                                          # Take Euler step, t_i -> t_{i+1}
             if sigma(t_{i+1}) != 0: # Not last
               d_i' <- (sigma' / sigma + s' / s)(t) * x_i - (sigma' * s / sigma)(t) D_theta(x_i / s(t) ; sigma(t)), t=t_{i+1} # Evaluate dx/dt at t_{i+1}
               x_{i+1} <- x_i + (t_{i+1} - t_i)(d_i + d_i') / 2
        3. Return: x_N
    """
    sde = self.sde
    current_t, next_t = t
    current_dsigma, current_sigma, current_s, current_ds = sde.heun_params(current_t) # 1, current_t, 1, 0
    next_dsigma, next_sigma, next_s, next_ds = sde.heun_params(next_t) # 1, next_t, 1, 0
    vec_current_t = current_sigma
    vec_next_t = next_sigma

    # First step
    denoise_current_t = self.score_fn(batch_mul(x, 1. / current_s), vec_current_t)
    d_i = batch_mul(current_dsigma / current_sigma + current_ds / current_s, x) - batch_mul(current_dsigma / current_sigma * current_s, denoise_current_t)
    x_prime_euler = x + batch_mul(next_t - current_t, d_i)

    # Second step unless next_t == 0
    denoise_next_t = self.score_fn(batch_mul(x_prime_euler, 1. / next_s), vec_next_t)
    dprime_i = batch_mul(next_dsigma / next_sigma + next_ds / next_s, x_prime_euler) - batch_mul(next_dsigma / next_sigma * next_s, denoise_next_t)
    x_prime_heun = x + batch_mul(next_t - current_t, (d_i + dprime_i) / 2)

    return x_prime_heun, x_prime_euler


@register_predictor(name='i2sb_solver')
class I2SBPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    sde = self.sde
    current_t, next_t = t

    score = self.score_fn(x, current_t) # epsilon_theta(x_t, t)
    noise = random.normal(rng, x.shape) # random normal
    c_signal, c_score, c_noise = sde.get_coeff(t) # Get coefficients from SDE specification.
    x_mean = batch_mul(c_signal, x) + batch_mul(c_score, score)
    if not self.probability_flow: # stochastic
      x    = x_mean + batch_mul(c_noise, noise)
    else:
      x    = x_mean
    return x, x_mean


@register_predictor(name='rf_solver')
class RFPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=True):
    super().__init__(sde, score_fn, probability_flow)
  
  def update_fn(self, rng, x, t):
    sde = self.sde
    current_t, next_t = t

    score = self.score_fn(x, current_t)
    x = x + batch_mul(score, next_t - current_t)
    return x, x


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, rng, x, t):
    return x, x

#############################################################################################################
@register_predictor(name='flow')
class FlowPredictor(Predictor):
  """
    Schrodinger bridge or Rectified flow predictor.
  """
  def __init__(self, sde, score_fn, latency, eps, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.model_fn = self.score_fn
    self.eps = eps
    self.sde = sde

  def rf_update_fn(self, rng, x, t):
    current_t, next_t = t # time in [eps, sde.T / 2]
    eps = self.eps
    sde = self.sde
    z, _ = self.model_fn(x, current_t * 999) # In RF, z is learned to return (x0 - x1) (old) and (x1 - x0) (new)
    diff_t = (next_t - current_t) * (sde.T - eps) / (sde.T / 2 - eps) # next_t - current_t, normalized < 0.
    x_mean = x + batch_mul(z, diff_t)
    return x_mean, x_mean
  
  def sb_update_fn(self, rng, x, t):
    current_t, next_t = t # time in [eps, sde.T / 2]
    z, _ = self.model_fn(x, current_t)
    pass

    return x, x_mean

  def update_fn(self, rng, x, t):
    if isinstance(self.sde, sde_lib.RFSDE):
      return self.rf_update_fn(rng, x, t)
    elif isinstance(self.sde, sde_lib.I2SBSDE):
      return self.sb_update_fn(rng, x, t)



@register_predictor(name='wavelet_ddim')
class WaveletDDIMPredictor(Predictor):
  """
    Wavelet DDIM predictor.
  """
  def __init__(self, sde, score_fn, latency, eps, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.latency = latency
    self.eps = eps
    self.sde = sde

  def update_fn(self, rng, x, t):
    current_t, next_t = t # time in [sde.T, sde.T * (3 / 2)]
    latency = self.latency # For example, 0.001 or 0.1.
    eps = self.eps
    sde = self.sde
    z = self.score_fn(x, current_t)

    x_low, x_high = haar_downsample(x)
    z_low, z_high = haar_downsample(z)

    # [1, 1.5] --> [0.5, 1]
    current_t, next_t = current_t - sde.T / 2, next_t - sde.T / 2
    # [0.5, 1] --> [latency, 1], interpolated linearly.
    current_t_low = jnp.maximum((latency - eps) / (sde.T / 2 - eps) * (current_t - sde.T / 2) + latency,
                                (sde.T - latency) / (sde.T / 2) * (current_t - sde.T) + sde.T)
    next_t_low = jnp.maximum((latency - eps) / (sde.T / 2 - eps) * (next_t - sde.T / 2) + latency,
                             (sde.T - latency) / (sde.T / 2) * (next_t - sde.T) + sde.T)

    current_alpha, current_sigma, _ = sde.snr(current_t_low)
    next_alpha, next_sigma, _ = sde.snr(next_t_low)
    A_signal = next_alpha / current_alpha
    A_score = current_sigma ** 2 * A_signal - current_sigma * next_sigma
    x_low = batch_mul(A_signal, x_low) + batch_mul(A_score, z_low)

    x_mean = haar_upsample(x_low, x_high)
    return x_mean, x_mean


#############################################################################################################
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      grad_norm = jnp.linalg.norm(
        grad.reshape((grad.shape[0], -1)), axis=-1).mean()
      grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
      noise_norm = jnp.linalg.norm(
        noise.reshape((noise.shape[0], -1)), axis=-1).mean()
      noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, rng, x, t):
    return x, x


def shared_predictor_update_fn(rng, state, x, t, sde, model, predictor, probability_flow, continuous, two_stage=False, latency=None, eps=None):
  """A wrapper that configures and returns the update function of predictors."""
  use_wavelet = False
  if not two_stage:
    score_fn = mutils.get_score_fn(sde, model, state.opt_state_ema.ema, state.states, train=False, continuous=continuous)
    if predictor.__name__ ==  'WaveletDDIMPredictor':
      use_wavelet = True
  else:
    score_fn = mutils.get_model_fn(model, state.opt_state_ema.ema, state.states, train=False)
    use_wavelet = True
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  elif use_wavelet:
    predictor_obj = predictor(sde, score_fn, latency, eps, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(rng, x, t)


def shared_corrector_update_fn(rng, state, x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, state.opt_state_ema.ema, state.states, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(rng, x, t)


def get_pc_sampler(sde, model, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, run_last_step=True, is_rf_sampler=False,
                   gen_reflow=False, reflow_t=1, soft_bound=0.0, timestep_style='denoising'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

    run_last_step: Flag to run last step (to t=0)
    is_rf_sampler: True - RF, False - PC
    gen_reflow: valid only if rf_sampler.
                Flag to generate reflow images
    reflow_t: Number of reflow time division. Default=1 (No division.)
    soft_bound: softening the bound, using the cosine filter. default=0.0

  Returns:
    A sampling function that takes random states, and a replcated training state and returns samples as well as
    the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          model=model,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  if is_rf_sampler:
    # Rectified flow sampler
    """
      Step 1. Define timesteps
        gen_reflow = True:  Reflow data generating phase. Should be conditional generation (should enter noiseless source data.)
        gen_reflow = False: 
      Step 2. Define sampler (named rf_sampler)
        Input
          rng: jax.random.PRNGKey
          state: flax.training.TrainState
          cond_image: None if not gen_reflow, noiseless source data if gen_reflow
        Return (gen_reflow)
          (
            inverse_scaler(x_mean if denoise else x), --> Generated image
            inverse_scaler(initial_image)             --> Noised initial image
          ),
          (
            vec_t_end,  --> ending time
            vec_t_start --> starting time
          )
        Return (not gen_reflow)
          inverse_scaler(x_mean if denoise else x),  --> Generated image
          stats                                      --> dictionary of statistics
    """
    # assertions
    assert (soft_bound >= 0.0) and (soft_bound <= 1.0)
    assert (reflow_t > 0) and (isinstance(reflow_t, int))

    # Get int-type variables outside pmap
    mod_t = int((sde.N - 1) // reflow_t)
    remainder = int((sde.N - 1) % reflow_t)
    n_div = jnp.ones([reflow_t], dtype=int) * int(mod_t) + jnp.concatenate([jnp.zeros([reflow_t - remainder]), jnp.ones(remainder)], dtype=int)
    cum_n_div = jnp.concatenate([jnp.zeros([1]), jnp.cumsum(n_div)])
    n_div, cum_n_div = n_div.tolist(), cum_n_div.tolist()
    cum_n_div = [int(c) for c in cum_n_div]
    cum_n_div[-1] += 1

    # Step 1. Define timesteps
    def get_timesteps(rng):
      """
        Input
          rng: jax.random.PRNGKey
        Return
          current_vec_t: (sde.N, B) array of 'current' time
          next_vec_t:    (sde.N, B) array of 'next' time

        Description.
          Return (current time, next time) for sampling, with shape (B,)
          The time steps style.
          Choice 1.
            linspace of [sde.T, eps] with sde.N - 1 intervals, and [eps, 0] with 1 interval (denoising)
              or
            linspace of [sde.T, 0] with sde.N interval (uniform or distillation)
      """
      rng, step_rng = random.split(rng)

      if gen_reflow:
        # Reflow data preprocessing phase
        rng, step_rng = random.split(rng)
        t_type = random.randint(step_rng, (shape[0],), minval=0, maxval=reflow_t)
        begin_t_hard = (t_type + 1) / reflow_t * sde.T
        end_t_hard = t_type / reflow_t * sde.T

        rng, step_rng = random.split(rng)
        cos_reg = (- jnp.cos(jnp.pi * random.uniform(step_rng, (shape[0],))) + 1.) / 2. # [0, 1]
        begin_t = jnp.minimum(begin_t_hard + soft_bound * cos_reg / (2 * reflow_t) * sde.T, sde.T) # (B,)
        end_t = jnp.maximum(end_t_hard - soft_bound * cos_reg / (2 * reflow_t) * sde.T, eps) # (B,)

        current_vec_t = jnp.linspace(begin_t, end_t, sde.N)
        next_vec_t = jnp.concatenate([current_vec_t[1:], current_vec_t[-1:] - (current_vec_t[-1:] == eps) * eps], axis=0)

      else:
        if timestep_style == 'denoising':
          # Sampling phase
          vec_t = []
          for i in range(reflow_t):
            t_max = (reflow_t - i) / reflow_t * sde.T
            t_min = jnp.maximum((reflow_t - i - 1) / reflow_t * sde.T, eps)
            if i == reflow_t - 1:
              vec_t.append(jnp.linspace(t_max, t_min, n_div[i] + 1))
            else:
              vec_t.append(jnp.linspace(t_max, t_min, n_div[i] + 1)[:-1])
            
          vec_t.append(jnp.zeros([1]))
          vec_t = jnp.concatenate(vec_t)

        elif timestep_style == 'distillation' or timestep_style == 'uniform': # do the same, but for visibility.
          for i in range(reflow_t):
            vec_t = jnp.linspace(sde.T, 0, sde.N + 1)
        else:
          raise NotImplementedError("timestep_style should be `denoising` or `distillation`.")

        current_vec_t = jnp.repeat(jnp.expand_dims(vec_t[:-1], axis=1), repeats=shape[0], axis=1)
        next_vec_t = jnp.repeat(jnp.expand_dims(vec_t[1:], axis=1), repeats=shape[0], axis=1)

      return current_vec_t, next_vec_t

    # Step 2. Define sampler
    def rf_sampler(rng, state, cond_image=None):

      rng, step_rng = random.split(rng)
      current_vec_t, next_vec_t = get_timesteps(step_rng)

      def loop_body(i, val):
        rng, x, x_mean, curv_diff = val
        vec_t = current_vec_t[i], next_vec_t[i]
        rng, step_rng = random.split(rng)
        new_x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)
        curv_diff.append(jnp.expand_dims(new_x - x, axis=0))
        curv_diff.pop(0)
        return rng, new_x, x_mean, curv_diff

      curv_diff = [jnp.zeros([1, *shape])] * sde.N # Dummy definition of curvatures

      if gen_reflow:
        rng, step_rng = random.split(rng)
        noise = sde.prior_sampling(step_rng, shape)
        initial_image = batch_mul(1. - current_vec_t[0], cond_image) + batch_mul(current_vec_t[0], noise)
        _, x, x_mean, _ = jax.lax.fori_loop(0, sde.N, loop_body, (rng, initial_image, initial_image, curv_diff))
      else:
        rng, step_rng = random.split(rng)
        initial_image = sde.prior_sampling(step_rng, shape)
        x = initial_image
        mid_images = [x]
        for rf_div in range(reflow_t):
          _, x, x_mean, curv_diff = jax.lax.fori_loop(int(cum_n_div[rf_div]), int(cum_n_div[rf_div + 1]), loop_body, (rng, x, x, curv_diff))
          mid_images.append(jax.lax.stop_gradient(x))

      # Calculate statistics using curvature statistics, if required.
      stats = dict()
      if not gen_reflow:
        curv_diff = jnp.concatenate(curv_diff, axis=0)      # array of (x_t' - x_t), (sde.N, B, H, W, C)
        t_diff = next_vec_t - current_vec_t                 # array of (t' - t),     (sde.N, B)
        lambda_mult = lambda a, b: a * b
        curv_derivative = jax.vmap(jax.vmap(lambda_mult, (0, 0), 0), (1, 1), 1)(curv_diff, 1. / t_diff)
        
        marginal_diff = x - initial_image                   # x_0 - x_1,             (B, H, W, C)
        straightness_gap = jnp.sum(jnp.square(- marginal_diff - curv_derivative), axis=(2, 3, 4)) # || (x_1 - x_0) - d/dt x_t  ||_2^2, (sde.N, B)
        straightness = jnp.mean(jnp.sum(batch_mul(- t_diff, straightness_gap), axis=0)) # (1,)
        straightness_by_t = jnp.mean(straightness_gap, axis=1) # (sde.N)

        mid_images = jnp.concatenate([jnp.expand_dims(m, axis=0) for m in mid_images], axis=0) # (reflow_t + 1, B, H, W, C)
        seq_diff = mid_images[1:] - mid_images[:-1] # (reflow_t, B, H, W, C)
        seq_straightness_gap = jnp.zeros((0, shape[0]))
        for r in range(reflow_t):
          mid_marginal_diff = seq_diff[r] * reflow_t                                              # (B, H, W, C)
          mid_curv_derivative = curv_derivative[cum_n_div[r]:cum_n_div[r + 1]]                    # (div[r], B, H, W, C)
          part_gap = jnp.sum(jnp.square(- mid_marginal_diff - mid_curv_derivative), axis=(2, 3, 4)) # (n_div[r], B)
          seq_straightness_gap = jnp.concatenate([seq_straightness_gap, part_gap], axis=0)        # finally (sde.N, B)
        
        assert seq_straightness_gap.shape[0] == sde.N
        seq_straightness = jnp.mean(jnp.sum(batch_mul(- t_diff, seq_straightness_gap), axis=0)) # (1,)
        seq_straightness_by_t = jnp.mean(seq_straightness_gap, axis=1) # (sde.N,)

        stats['straightness'] = straightness
        stats['straightness_by_t'] = straightness_by_t
        stats['seq_straightness'] = seq_straightness
        stats['seq_straightness_by_t'] = seq_straightness_by_t

        del curv_diff, t_diff, curv_derivative, marginal_diff, mid_images, seq_diff, seq_straightness_gap, part_gap, mid_marginal_diff, mid_curv_derivative

      return ((
        inverse_scaler(x_mean if denoise else x), inverse_scaler(initial_image)
      ), (
        next_vec_t[-1:], # ending point
        current_vec_t[0:1], # starting point
      ),
      stats)

    return jax.pmap(rf_sampler, axis_name='batch')

  else:
    # Usual PC sampler
    def pc_sampler(rng, state, cond_image=None):
      """ The PC sampler funciton.

      Args:
        rng: A JAX random state
        state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
        cond_image: A `jnp.array` object, input image for conditional generation.
        last_step_denoising: True if denoising for last step, False otherwise.
      Returns:
        Samples, number of function evaluations
      """
      # Initial sample
      rng, step_rng = random.split(rng)
      if cond_image is None:
        x = sde.prior_sampling(step_rng, shape)
      else: # I2SB case
        x = cond_image

      initial_image = x

      if isinstance(sde, sde_lib.EDMSDE):
        # Timesteps for second-order Heun sampler
        # timesteps: (((N - 1 - t) * sigma_max ** (1/rho) + t * (sigma_min ** (1/rho)) / (N - 1)) ** rho
        ts_idx = jnp.arange(sde.N, dtype=jnp.float32) # (0, 1, ...., N-1)
        timesteps = (sde.sigma_max ** (1. / sde.rho) + ts_idx / (sde.N - 1) * (sde.sigma_min ** (1. / sde.rho) - sde.sigma_max ** (1. / sde.rho))) ** sde.rho
        next_timesteps = jnp.concatenate([timesteps[1:], jnp.array([0.])]) # last: noise (sigma_min --> zero).
      else:
        timesteps = jnp.linspace(sde.T, eps, sde.N)
        next_timesteps = jnp.concatenate([timesteps[1:], jnp.zeros([1])])

      if not run_last_step:
        timesteps, next_timesteps = timesteps[:-1], next_timesteps[:-1] # Remove last step.

      def loop_body(i, val):
        rng, x, x_mean = val
        vec_t = jnp.ones(shape[0]) * timesteps[i], jnp.ones(shape[0]) * next_timesteps[i]
        rng, step_rng = random.split(rng)
        x, x_mean = corrector_update_fn(step_rng, state, x, vec_t)
        rng, step_rng = random.split(rng)
        x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)
        return rng, x, x_mean

      _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
      # Denoising is equivalent to running one predictor step without adding noise.
      return (inverse_scaler(x_mean if denoise else x), initial_image), sde.N * (n_steps + 1)

    return jax.pmap(pc_sampler, axis_name='batch')


# def get_ode_sampler(sde, model, shape, inverse_scaler,
#                     denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3):
def get_ode_sampler(sde, model, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.

  Returns:
    A sampling function that takes random states, and a replicated training state and returns samples
    as well as the number of function evaluations during sampling.
  """

  @jax.pmap
  def denoise_update_fn(rng, state, x):
    score_fn = get_score_fn(sde, model, state.opt_state_ema.ema, state.states, train=False, continuous=True)
    if isinstance(sde, sde_lib.RFSDE):
      vec_eps = jnp.ones((x.shape[0],)) * eps
      x = x - batch_mul(score_fn(x, vec_eps, rng), vec_eps)
    else:
      # Reverse diffusion predictor for denoising
      predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
      vec_eps = jnp.ones((x.shape[0],)) * eps
      _, x = predictor_obj.update_fn(rng, x, vec_eps)
    return x

  @jax.pmap
  def drift_fn(state, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, state.opt_state_ema.ema, state.states, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(prng, pstate, z=None, compute_straightness=False):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      prng: An array of random state. The leading dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      z: If present, generate samples from latent code `z`.
    Returns:
      Samples, and the number of function evaluations.
    """
    # Initial sample
    rng = flax.jax_utils.unreplicate(prng)
    rng, step_rng = random.split(rng)
    if z is None:
      # If not represent, sample the latent code from the prior distibution of the SDE.
      x = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape)
      noise = x
    else:
      x = jnp.reshape(z, (jax.local_device_count(), z.shape[0] // jax.local_device_count(), *z.shape[1:]))
      noise = x

    def ode_func(t, x):
      x = from_flattened_numpy(x, (jax.local_device_count(),) + shape)
      vec_t = jnp.ones((x.shape[0], x.shape[1])) * t
      drift = drift_fn(pstate, x, vec_t)
      return to_flattened_numpy(drift)

    # Black-box ODE solver for the probability flow ODE
    solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                   rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    x = jnp.asarray(solution.y[:, -1]).reshape((jax.local_device_count(),) + shape)

    # Denoising is equivalent to running one predictor step without adding noise
    if denoise:
      rng, *step_rng = random.split(rng, jax.local_device_count() + 1)
      step_rng = jnp.asarray(step_rng)
      x = denoise_update_fn(step_rng, pstate, x)

    x = inverse_scaler(x)
    return (x, noise), nfe

  return ode_sampler


def get_two_stage_sampler(sde, model, shape, predictor_high, predictor_low, inverse_scaler, n_steps,
                          denoise, latency, eps, probability_flow=False, continuous=False):
  predictor_high_update_fn = functools.partial(shared_predictor_update_fn,
                                               sde=sde["high"],
                                               model=model,
                                               predictor=predictor_high,
                                               probability_flow=probability_flow,
                                               latency=latency,
                                               eps=eps,
                                               two_stage=False, # to use get_score_fn
                                               continuous=continuous)
  predictor_low_update_fn = functools.partial(shared_predictor_update_fn,
                                              sde=sde["low"],
                                              model=model,
                                              predictor=predictor_low,
                                              probability_flow=probability_flow,
                                              latency=latency,
                                              eps=eps,
                                              two_stage=True,
                                              continuous=continuous)

  def two_stage_sampler(rng, state):
    # initial sample
    rng, step_rng = random.split(rng)
    x = sde["high"].prior_sampling(step_rng, shape) # Sample uniform from VP.

    timestep_high_res = jnp.linspace(sde["high"].T / 2, sde["high"].T, sde["high"].N + 1) + sde["high"].T / 2 # [sde.T, sde.T * 1.5]

    timesteps_high_res_current = timestep_high_res[1:]
    timesteps_high_res_next = timestep_high_res[:-1]
    timesteps_high_res_current = jnp.flip(timesteps_high_res_current) # sde.T --> sde.T / 2, current
    timesteps_high_res_next = jnp.flip(timesteps_high_res_next) # sde.T --> sde.T / 2, next

    def loop_body_high(i, val):
      rng, x, x_mean = val
      t_current = timesteps_high_res_current[i]
      vec_t_current = jnp.ones(shape[0]) * t_current
      t_next = timesteps_high_res_next[i]
      vec_t_next = jnp.ones(shape[0]) * t_next
      vec_t = (vec_t_current, vec_t_next)
      rng, step_rng = random.split(rng)
      x, x_mean = predictor_high_update_fn(step_rng, state, x, vec_t)
      return rng, x, x_mean
    
    timesteps_low_res_current = jnp.linspace(eps, sde["low"].T / 2, sde["low"].N) # [eps, sde.T * 0.5]
    timesteps_low_res_next = jnp.concatenate([jnp.array([0]), timesteps_low_res_current[:-1]], axis=0)
    timesteps_low_res_current = jnp.flip(timesteps_low_res_current) # sde.T / 2 --> eps, current
    timesteps_low_res_next = jnp.flip(timesteps_low_res_next) # sde.T / 2 --> eps, next

    def loop_body_low(i, val):
      rng, x, x_mean = val
      t_current = timesteps_low_res_current[i]
      vec_t_current = jnp.ones(shape[0]) * t_current
      t_next = timesteps_low_res_next[i]
      vec_t_next = jnp.ones(shape[0]) * t_next
      vec_t = (vec_t_current, vec_t_next)
      rng, step_rng = random.split(rng)
      x, x_mean = predictor_low_update_fn(step_rng, state, x, vec_t)
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde["high"].N, loop_body_high, (rng, x, x))
    _, x, x_mean = jax.lax.fori_loop(0, sde["low"].N, loop_body_low, (rng, x, x))
    return inverse_scaler(x_mean if denoise else x), sde["high"].N + sde["low"].N
    
  return jax.pmap(two_stage_sampler, axis_name='batch')