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
"""Utility code for generating and saving image grids and checkpointing.

   The `save_image` code is copied from
   https://github.com/google/flax/blob/master/examples/vae/utils.py,
   which is a JAX equivalent to the same function in TorchVision
   (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
"""

import math
from typing import Any, Dict, Optional, TypeVar

import flax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import tensorflow as tf
import os

import evaluation
import gc
import io
import logging
import tensorflow_gan as tfgan

from jax.experimental.host_callback import call
import datasets

T = TypeVar("T")


def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


def load_training_state(filepath, state):
  with tf.io.gfile.GFile(filepath, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
  """Make a grid of images and save it into an image file.

  Pixel values are assumed to be within [0, 1].

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """
  if not (isinstance(ndarray, jnp.ndarray) or isinstance(ndarray, np.ndarray) or
          (isinstance(ndarray, list) and
           all(isinstance(t, jnp.ndarray) for t in ndarray))):
    raise TypeError("array_like of tensors expected, got {}".format(
      type(ndarray)))

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                       padding)
  num_channels = ndarray.shape[3]
  grid = jnp.full(
    (height * ymaps + padding, width * xmaps + padding, num_channels),
    pad_value).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width].set(ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
  im = Image.fromarray(np.array(ndarr.copy()))
  im.save(fp, format=format)


def flatten_dict(config):
  """Flatten a hierarchical dict to a simple dict."""
  new_dict = {}
  for key, value in config.items():
    if isinstance(value, dict):
      sub_dict = flatten_dict(value)
      for subkey, subvalue in sub_dict.items():
        new_dict[key + "/" + subkey] = subvalue
    elif isinstance(value, tuple):
      new_dict[key] = str(value)
    else:
      new_dict[key] = value
  return new_dict


def to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x.reshape((-1,)))


def from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x).reshape(shape)


def draw_figure_grid(sample, sample_dir, figname):
  """Draw grid of figures; samples are of [0, 1]-valued numpy arrays."""
  tf.io.gfile.makedirs(sample_dir)
  image_grid = sample.reshape((-1, *sample.shape[-3:]))
  nrow = int(np.sqrt(image_grid.shape[0]))
  sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
  with tf.io.gfile.GFile(
      os.path.join(sample_dir, f"{figname}.np"), "wb") as fout:
    np.save(fout, sample)

  with tf.io.gfile.GFile(
      os.path.join(sample_dir, f"{figname}.png"), "wb") as fout:
    save_image(image_grid, fout, nrow=nrow, padding=2)


def get_samples_and_statistics(config, rng, sampling_fn, pstate, sample_dir, sample_shape, mode='train', save_samples=False):
  """
    Sampling pipeline, including statistics
  """
  if mode == 'train':
    # train mode: snapshot sampling
    n_samples = config.training.snapshot_fid_sample
    b_size = config.eval.batch_size
  elif mode == 'eval':
    # eval mode: sampling for evaluation
    n_samples = config.eval.num_samples
    b_size = config.eval.batch_size
  else:
    raise NotImplementedError()
  num_sampling_rounds = (n_samples - 1) // b_size + 1
  tf.io.gfile.makedirs(sample_dir)

  sequential_rf = (config.training.reflow_t > 1) and (config.model.num_scales > 1) # TODO: Make this condition clearer

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  # Sample
  all_pools = []
  all_straightness_dict = {
    'straightness': [],
    'straightness_by_t': [],
    'seq_straightness': [],
    'seq_straightness_by_t' : [],
  }

  for i in range(num_sampling_rounds):
    logging.info(f"Round {i + 1} for sampling.")
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
    if config.sampling.predictor == 'rf_solver':
      (samples, z), _, straightness = sampling_fn(sample_rng, pstate)
      with tf.io.gfile.GFile(
          os.path.join(sample_dir, f"straightness_{i+1}.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer,
                            straightness=straightness['straightness'],
                            straightness_by_t=straightness['straightness_by_t'],
                            seq_straightness=straightness['seq_straightness'],
                            seq_straightness_by_t=straightness['seq_straightness_by_t'])
        fout.write(io_buffer.getvalue())
    else:
      # TODO
      raise NotImplementedError()
      (samples, z), nfe = sampling_fn(sample_rng, pstate)

    gc.collect()
    samples = samples.reshape((-1, *samples.shape[-3:]))

    # Visualize example images in first step
    if i == 0:
      image_grid = samples[0:64]
      draw_figure_grid(image_grid, sample_dir, "sample")

    # Save images to `samples_*.npz`
    samples_save = np.clip(samples * 255., 0, 255).astype(np.uint8) # [0, 1] --> [0, 255]
    with tf.io.gfile.GFile(
        os.path.join(sample_dir, f"samples_{i+1}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=samples_save)
      fout.write(io_buffer.getvalue())

    # Save stats to `statistics_*.npz
    # Force garbage collection before calling TensorFlow code for Inception network
    gc.collect()
    latents = evaluation.run_inception_distributed(samples_save, inception_model,
                                                    inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()
    # Save latent represents of the Inception network to disk or Google Cloud Storage
    with tf.io.gfile.GFile(
        os.path.join(sample_dir, f"statistics_{i+1}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(
        io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
      fout.write(io_buffer.getvalue())

  # Check if there is pretrained inception pool layer statistics
  data_stats, have_stats = evaluation.load_dataset_stats(config)
  if have_stats:
    data_pools = data_stats["pool_3"]

  else:
    # Build training dataset iterators.
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=config.training.n_jitted_steps,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

    # Newly generate dataset statistics.
    train_pools = []
    if not inceptionv3:
      train_logits = []

    train_iter = iter(train_ds)
    for i, batch in enumerate(train_iter):
      train_batch = jax.tree_util.tree_map(lambda x: x._numpy(), batch)
      train_batch_resize = jax.image.resize(train_batch['image'],
                                            (*train_batch['image'].shape[:-3], *sample_shape[-3:]),
                                            method='nearest')
      train_batch_int = np.clip(train_batch_resize * 255., 0, 255).astype(np.uint8)
      train_batch_images = train_batch_int.reshape((-1, *train_batch_int.shape[-3:]))
      train_latents = evaluation.run_inception_distributed(train_batch_images, inception_model,
                                                            inceptionv3=inceptionv3)
      train_pools.append(train_latents['pool_3'])
      if not inceptionv3:
        train_logits.append(train_latents['logits'])
    data_pools = jnp.array(train_pools).reshape(-1, train_pools[0].shape[-1])
    if not inceptionv3:
      data_logits = jnp.array(train_logits).reshape(-1, train_logits[0].shape[-1])
    
    if not inceptionv3:
      np.savez_compressed(data_stats, pool_3=data_pools, logits=data_logits)
    else:
      np.savez_compressed(data_stats, pool_3=data_pools)

  # Compute statistics (FID/KID/IS/straightness)
  all_logits = []
  all_pools = []
  stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
  wait_message = False
  while len(stats) < num_sampling_rounds:
    if not wait_message:
      logging.warning("Waiting for statistics on host %d" % (host,))
      wait_message = True
    stats = tf.io.gfile.glob(
      os.path.join(sample_dir, "statistics_*.npz"))
    time.sleep(30)

  for stat_file in stats:
    with tf.io.gfile.GFile(stat_file, "rb") as fin:
      stat = np.load(fin)
      if not inceptionv3:
        all_logits.append(stat["logits"])
      all_pools.append(stat["pool_3"])

  if not inceptionv3:
    all_logits = np.concatenate(
      all_logits, axis=0)[:n_samples]
  all_pools = np.concatenate(all_pools, axis=0)[:n_samples]

  if not inceptionv3:
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
  else:
    inception_score = -1

  fid = tfgan.eval.frechet_classifier_distance_from_activations(
    data_pools, all_pools)
  # Hack to get tfgan KID work for eager execution.
  tf_data_pools = tf.convert_to_tensor(data_pools)
  tf_all_pools = tf.convert_to_tensor(all_pools)
  kid = tfgan.eval.kernel_classifier_distance_from_activations(
    tf_data_pools, tf_all_pools).numpy()
  del tf_data_pools, tf_all_pools
  gc.collect()

  # Return values
  stats_dict = dict()
  stats_dict["is"] = inception_score
  stats_dict["fid"] = fid
  stats_dict["kid"] = kid
  if config.sampling.predictor == 'rf_solver':
    straightness_files = tf.io.gfile.glob(os.path.join(sample_dir, "straightness_*.npz"))
    for straightness_file in straightness_files:
      with tf.io.gfile.GFile(straightness_file, "rb") as fin:
        straightness = np.load(straightness_file)
        for k in all_straightness_dict:
          all_straightness_dict[k].append(straightness[k])

    all_straightness = dict()
    for k in all_straightness_dict:
      all_straightness[k] = jnp.mean(jnp.concatenate([jnp.expand_dims(v_, axis=0) for v_ in all_straightness_dict[k]], axis=0), axis=(0, 1))
    stats_dict["straightness"] = all_straightness

  _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))]
  _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(sample_dir, "straightness_*.npz"))]
  del inception_model

  if not save_samples:
    _ = [tf.io.gfile.remove(f) for f in tf.io.gfile.glob(os.path.join(sample_dir, "samples_*.npz"))] # remove samples
  return stats_dict

def jprint(*args):
  fstring = ""
  arrays = []
  for i, a in enumerate(args):
    if i != 0:
      fstring += " "
    if isinstance(a, str):
      fstring += a
    else:
      fstring += '{}'
      arrays.append(a)

  call(lambda arrays: print(fstring.format(*arrays)), arrays)