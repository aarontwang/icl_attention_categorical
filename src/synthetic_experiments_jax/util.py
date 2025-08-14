import os
import io

from typing import Any, MutableMapping, NamedTuple, Tuple

import jax
from jax import grad, jit, vmap
import jax.numpy as jnp

from tensorflow import keras
from keras.applications.vgg16 import VGG16

import haiku as hk
import math

from config import config
from train import *
from data import create_img_data, get_train_features, get_val_features


def conf_init(save_path, num_seeds, num_layers, recurrent, data_kernel, model_kernel, cat, use_bias_head, use_bias_data,
              gamma=None, sigma=None, c_size=100, i_size=2, e_size=5, j=5, k=5, l=3, dist=0.1, one_hot_emb=False,
              num_heads_gd=1, num_heads_tr=1, gd_plus=False, diag=False, early_stopping=False, unique_w_e=False,
              examples_per_cat=10, training_data_size=2048, minibatch_size=512, training_steps_tf=5000,
              training_steps_gd=5000,
              hold_out=False, gd_init=False, linear_approx=False, use_mlp=False, num_queries=1):
    recurrent_transformer = recurrent
    num_layers = num_layers
    num_seeds = num_seeds
    config.save_folder = save_path

    config.cats = cat
    config.output_size = e_size
    config.e_size = cat-1 if one_hot_emb else e_size
    config.mlp_output_dim = e_size
    config.second_layer = False
    config.widening_factor = 10

    config.bias_head = use_bias_head
    config.bias_data = use_bias_data
    config.one_hot_emb = one_hot_emb

    config.data_e_size = e_size
    config.unique_w_e = unique_w_e
    config.num_queries = num_queries

    config.early_stopping = early_stopping

    if config.bias_head:
        num_heads_gd = 2

    config.gd_plus = gd_plus
    config.diag = diag

    config.seed = 0
    config.local_usage = True

    config.gd_init = gd_init

    ####
    config.deq = recurrent_transformer
    config.gd_deq = recurrent_transformer
    config.att_only_trans = True
    ####

    config.pre_train_gd = True
    config.train_gd_whitening = True
    config.train_gd_lr = True
    config.use_bias = False
    config.include_query = False

    # for kernel
    config.data_kernel = data_kernel
    config.model_kernel = model_kernel
    config.j = 5 if j is None else j
    config.k = 5 if k is None else k
    config.l = 3 if l is None else l
    config.dist = 0.1 if dist is None else dist
    config.lam = 1e5
    config.gamma = 1 if gamma is None else gamma
    config.sigma = 1 if sigma is None else sigma

    config.training_steps = training_steps_tf
    config.training_steps_gd_constructed = training_steps_gd
    config.training_steps_linear = 500

    config.layer_norm = False
    config.out_proj = False
    config.in_proj = True
    config.adam = True
    config.dataset_size = c_size
    config.input_size = i_size
    config.key_size = config.input_size + 3 * config.e_size
    config.num_layers = num_layers
    config.num_heads_tr = num_heads_tr
    config.num_heads_gd = num_heads_gd
    config.linear_approx = linear_approx
    config.use_mlp = use_mlp

    config.top_preds = 2 if data_kernel == 'imagenet' else 3

    config.grad_clip_value = 10 if num_layers > 2 else 0.001
    config.grad_clip_value_gd = 10 if num_layers > 2 else 0.001

    config.gd_transition_steps = 1000
    config.tf_transition_steps = 500
    config.gd_lr_decay_rate = 0.8
    config.lr_decay_rate = 0.5
    config.transition_begin = 0
    config.staircase = True

    if gd_init:
        config.tf_use_lr_schedule = True
    else:
        config.tf_use_lr_schedule = True if num_layers >= 2 else False

    config.gd_use_lr_schedule = True if num_layers >= 2 else False

    if config.model_kernel == 'rbf':
        if data_kernel == 'random_grid':
            config.lr = 0.001
        elif data_kernel == 'high_dim':
            if gd_init:
                config.lr = 0.00001
            else:
                if num_layers == 1:
                    config.lr = 0.001
                else:
                    config.lr = 0.00001
        else:
            config.lr = 0.001
    elif config.model_kernel == 'softmax':
        if data_kernel == 'random_grid':
            config.lr = 0.001
        elif data_kernel == 'high_dim':
            if gd_init:
                config.lr = 0.00001
            else:
                if num_layers == 1:
                    config.lr = 0.001
                else:
                    config.lr = 0.0005
                    config.lr_decay_rate = 0.9
                    config.tf_transition_steps = 1000
                    config.tf_use_lr_schedule = False
        else:
            config.lr = 0.001
    elif config.model_kernel == 'linear':
        if data_kernel == 'random_grid':
            config.lr = 0.001
        elif data_kernel == 'high_dim':
            if gd_init:
                config.lr = 0.00001
            else:
                if num_layers == 1:
                    config.lr = 0.001
                else:
                    config.lr = 0.0005
        else:
            config.lr = 0.001
            config.gd_use_lr_schedule = True
    elif config.model_kernel == 'exp':
        config.lr = 0.001
    elif config.model_kernel == 'laplacian':
        config.lr = 0.001

    if num_layers > 3:
        config.gd_lr = 0.001
    else:
        if config.model_kernel == 'rbf':
            if data_kernel == 'random_grid':
                config.gd_lr = 0.001
            elif data_kernel == 'high_dim':
                if num_layers == 1:
                    config.gd_lr = 0.001
                elif num_layers >= 2:
                    config.gd_lr = 0.0002
                    config.gd_lr_decay_rate = 0.85
            else:
                config.gd_lr = 0.001
        elif config.model_kernel == 'softmax':
            if data_kernel == 'random_grid':
                config.gd_lr = 0.001
            elif data_kernel == 'high_dim':
                if num_layers == 1:
                    if config.use_mlp:
                        config.gd_lr = 0.003
                    else:
                        config.gd_lr = 0.001
                elif num_layers >= 2:
                    config.gd_lr = 0.0005
                    config.gd_lr_decay_rate = 0.9
                    config.gd_transition_steps = 1000
                    config.gd_use_lr_schedule = False
            else:
                config.gd_lr = 0.001
        elif config.model_kernel == 'linear':
            if data_kernel == 'random_grid':
                config.gd_lr = 0.001
            elif data_kernel == 'high_dim':
                if num_layers == 1:
                    config.gd_lr = 0.001
                elif num_layers >= 2:
                    config.gd_lr = 0.0001
            else:
                if num_layers == 1:
                    config.gd_lr = 0.00005
                elif num_layers >= 2:
                    config.gd_lr = 0.001
        elif config.model_kernel == 'exp':
            config.gd_lr = 0.001
        elif config.model_kernel == 'laplacian':
            config.gd_lr = 0.001

    config.wd = 0.0
    config.init_scale = 0.002 / config.num_layers
    # config.init_scale = 1 / np.sqrt(config.key_size)

    if config.data_kernel == 'high_dim':
        config.bs_eval = 2048
        config.bs_train = training_data_size
    elif config.data_kernel == 'imagenet':
        config.bs_eval = 2048
        config.bs_train = training_data_size

    if minibatch_size > training_data_size:
        config.mb_size = training_data_size
    else:
        config.mb_size = minibatch_size

    config.dampening = 1.0
    config.gd_dampening = 1.0
    config.clip = 10 if num_layers > 3 else 0

    config.dropout_rate = 0.0

    config.input_range = 1

    config.analyze = True

    config.cycle_data = 0
    config.num_seeds = num_seeds

    if config.num_layers == 1:
        assert config.deq is True
        assert config.gd_deq is True

    config.in_proj = False

    # for ImageNet experiments
    config.examples_per_cat = examples_per_cat
    config.holdout = hold_out

    # ImageNet data
    config.model = VGG16(weights='imagenet', include_top=False)
    config.input_shape = [224, 224, 3]
    config.target_size = [224, 224]

    config.data_path = "train_features.npy"
    config.train_features = jnp.load("train_features.npy") if os.path.exists(
        "train_features.npy") else get_train_features()
    config.train_lens = jnp.load("train_lens.npy")
    config.train_min_len = 732

    if hold_out:
        rng = jax.random.PRNGKey(6)
        rng, val_rng = jax.random.split(rng, 2)

        hold_out_idxs = jax.random.choice(val_rng, 1000, shape=(100,), replace=False)
        mask = jnp.zeros(1000, dtype=bool).at[hold_out_idxs].set(True)

        # Split the array into two arrays
        config.val_features = config.train_features[mask]
        config.train_features = config.train_features[~mask]

        config.val_lens = config.train_lens[mask]
        config.val_min_len = 732
    else:
        config.val_features = jnp.load("val_features.npy") if os.path.exists("val_features.npy") else get_val_features()

        config.val_lens = jnp.load("val_lens.npy")
        config.val_min_len = 50

    config.model = None
