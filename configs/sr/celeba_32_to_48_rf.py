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
from configs.rf.celeba_32_rf_continuous import get_config as get_config_low


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'rfsde'
  training.continuous = True
  training.reduce_mean = True
  training.snapshot_sampling = False
  training.batch_size = 128
  training.n_reflow_data = 10000
  training.refresh_reflow_step = 10000 # number of steps that reflow batch is generated again.

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'rf_solver'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'iddpm'
  model.ema_rate = 0.9999
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 3
  model.attn_resolutions = (48, 24, 12,)
  model.fir_kernel = [1, 1]
  model.init_scale = 0.
  model.nonlinearity = 'swish'

  model.label_dim = 0
  model.augment_dim = 0
  model.label_dropout = 0.0
  model.channel_mult_emb = 4

  # Evaluation hyperparameters
  evaluate = config.eval
  evaluate.enable_loss = False
  evaluate.sample_low_resolution = False # True: Sampled low-res + SR, False: Low-res training set + SR
  evaluate.low_nfe = 100
  evaluate.low_ckpt = 9
  evaluate.ckpt_dir_low = 'exp/celeba_32_rf_ncsnpp'

  # RF hyperparameters
  model.rf_task = 'sr'
  model.rf_phase = 1
  model.from_res = 32
  model.to_res = 48
  model.high_sigma = 0.0

  # Low-resolution config
  low = config.low = get_config_low()
  low.run_last_step = True

  return config
