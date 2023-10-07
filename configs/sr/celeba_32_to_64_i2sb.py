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

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with VP SDE."""
from configs.default_celeba_configs import get_default_configs
from configs.edm.celeba_32_edm_continuous import get_config as get_config_low

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'i2sb'
  training.continuous = True
  training.reduce_mean = True
  training.snapshot_sampling = True
  training.snapshot_freq = 50000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'i2sb_solver'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'positional'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.num_scales = 50

  # I2SB hyperparameters
  model.beta_min = 0.1
  model.beta_max = 0.3
  model.from_res = 32
  model.to_res = 64
  model.high_sigma = 0.0

  # Evaluation hyperparameters
  evaluate = config.eval
  evaluate.enable_loss = False
  evaluate.sample_low_resolution = False # True: Sampled low-res + SR, False: Low-res training set + SR
  evaluate.low_nfe = 18
  evaluate.low_ckpt = 9
  evaluate.ckpt_dir_low = 'exp/celeba_32_edm_ncsnpp'
  
  # Low-resolution config
  low = config.low = get_config_low()
  low.run_last_step = True

  return config
