import os
import io

from typing import Any, MutableMapping, NamedTuple, Tuple

from tensorflow import keras
from keras.applications.vgg16 import VGG16

import math

from config import config


def conf_init(save_path, model_type, num_seeds, num_layers, data_kernel, model_kernel, num_cats,
              c_size=100, i_size=2, e_size=5, high_dim_k=5, high_dim_lambda=3, 
              dist=0.1, mixture_j=10, mixture_u_var=5,
              one_hot_emb=False, gd_plus=False, unique_w_e=False,
              examples_per_cat=10, num_ff_layers=1, ff_hidden_size=10, num_heads=1,
              num_mlps=1, moe_k=1, shared_ff=False,
              training_data_size=2048, minibatch_size=512,
              training_steps_tf=5000, training_steps_gd=5000,
              holdout=True, gd_init=False, use_mlp=False, num_queries=1):
    config.save_folder = save_path  # directory to save results
    config.num_seeds = num_seeds  # number of seeds to run on

    config.seed = 0  # current seed

    # trained tf params
    config.model_type = model_type  # interleaved or linear approximation
    config.model_kernel = model_kernel  # kernel to use in transformer (linear, RBF, softmax)
    config.num_layers = num_layers  # number of layers in transformer

    config.num_queries = num_queries  # number of queries per contextual set, default is 1

    config.include_query = False  # whether to include the queries in the key and value inputs of the transformer

    config.use_mlp = use_mlp  # whether to use feedforward network in transformer
    config.gd_init = gd_init  # whether to initialize Trained TF with GD weights
    config.one_hot_emb = one_hot_emb  # whether to use one-hot embeddings for transformer embedding matrix
    config.num_ff_layers = num_ff_layers
    config.ff_hidden_size = ff_hidden_size
    config.shared_ff = shared_ff
    config.num_heads = num_heads
    config.num_mlps = num_mlps
    config.moe_k = moe_k

    config.lam = 1e5  # scaling factor of erasure head in self-attention layer of interleaved attention

    # dimension of input covariates x_i
    config.input_size = i_size
    config.e_size = num_cats - 1 if one_hot_emb else e_size  # dimension of embeddings for each category
    config.mlp_output_dim = e_size  # output dimension of transformer feedforward element
    config.second_layer = False
    config.widening_factor = 10

    config.num_cats = num_cats  # number of categories
    config.output_size = e_size  # output of transformer attention layers

    config.gd_plus = gd_plus  # whether to use GD++

    config.layer_norm = False
    config.out_proj = False
    config.in_proj = True

    config.init_scale = 0.002 / config.num_layers  # scale factor for initialization of transformer parameters

    # data generation parameters
    config.data_kernel = data_kernel  # high-dimensional or imagenet

    # high-dimensional data generation parameters
    config.high_dim_k = 5 if high_dim_k is None else high_dim_k
    config.high_dim_lambda = 3 if high_dim_lambda is None else high_dim_lambda
    config.dist = 0.1 if dist is None else dist
    config.input_range = 1

    config.mixture_j = mixture_j
    config.mixture_u_var = mixture_u_var

    config.data_e_size = e_size  # dimension of embedding vectors for data generation
    config.unique_w_e = unique_w_e  # whether to use

    config.dataset_size = c_size  # number of contextual examples in each contextual dataset

    # size of each minibatch
    if minibatch_size > training_data_size:
        config.mb_size = training_data_size
    else:
        config.mb_size = minibatch_size

    config.bs_eval = 2048  # number of contextual datasets for validation/evaluation
    config.bs_train = training_data_size  # number of contextual datasets for training

    config.top_preds = 2 if data_kernel == 'imagenet' else 3

    # transformer training parameters
    config.training_steps_tf = training_steps_tf  # number of training steps for Trained TF
    config.training_steps_gd = training_steps_gd  # number of training steps for GD
    config.training_steps_linear = 500  # number of training steps for linear probing

    # whether to use gradient clipping for GD and Trained TF
    config.grad_clip_value_tf = 10 if num_layers > 2 else 0.001
    config.grad_clip_value_gd = 10 if num_layers > 2 else 0.001

    config.wd = 0.0  # weight decay for AdamW optimizer
    config.clip = 10 if num_layers > 3 else 0

    # whether to use learning rate scheduler for Trained TF
    if gd_init:
        config.tf_use_lr_schedule = True
    else:
        config.tf_use_lr_schedule = True if num_layers >= 2 else False

    # whether to use learning rate scheduler for GD
    config.gd_use_lr_schedule = True if num_layers >= 2 else False

    # learning rate schedule parameters
    config.gd_transition_steps = 1000
    config.tf_transition_steps = 500
    config.gd_lr_decay_rate = 0.8
    config.lr_decay_rate = 0.5
    config.transition_begin = 0
    config.staircase = True

    # Trained TF learning rates
    config.lr = 0.001
    # if config.model_kernel == 'rbf':
    #     if data_kernel == 'random_grid':
    #         config.lr = 0.001
    #     elif data_kernel == 'high_dim':
    #         if gd_init:
    #             config.lr = 0.00001
    #         else:
    #             if num_layers == 1:
    #                 config.lr = 0.001
    #             else:
    #                 config.lr = 0.00001
    #     else:
    #         config.lr = 0.001
    # elif config.model_kernel == 'softmax':
    #     if data_kernel == 'random_grid':
    #         config.lr = 0.001
    #     elif data_kernel == 'high_dim':
    #         if gd_init:
    #             config.lr = 0.00001
    #         else:
    #             if num_layers == 1:
    #                 config.lr = 0.001
    #             else:
    #                 config.lr = 0.0005
    #                 config.lr_decay_rate = 0.9
    #                 config.tf_transition_steps = 1000
    #                 config.tf_use_lr_schedule = False
    #     elif data_kernel == 'high_dim_mixture':
    #         config.lr = 0.001
    #     else:
    #         config.lr = 0.001
    # elif config.model_kernel == 'linear':
    #     if data_kernel == 'random_grid':
    #         config.lr = 0.001
    #     elif data_kernel == 'high_dim':
    #         if gd_init:
    #             config.lr = 0.00001
    #         else:
    #             if num_layers == 1:
    #                 config.lr = 0.001
    #             else:
    #                 config.lr = 0.0005
    #     else:
    #         config.lr = 0.001
    #         config.gd_use_lr_schedule = True
    config.lr_decay_rate = 0.9
    config.tf_transition_steps = 1000
    config.tf_use_lr_schedule = False
    config.gd_use_lr_schedule = False

    # GD learning rates
    config.gd_lr = 0.001
    # if num_layers > 3:
    #     config.gd_lr = 0.001
    # else:
    #     if config.model_kernel == 'rbf':
    #         if data_kernel == 'high_dim':
    #             if num_layers == 1:
    #                 config.gd_lr = 0.001
    #             elif num_layers >= 2:
    #                 config.gd_lr = 0.0002
    #                 config.gd_lr_decay_rate = 0.85
    #         elif data_kernel == 'imagenet':
    #             config.gd_lr = 0.01
    #     elif config.model_kernel == 'softmax':
    #         if data_kernel == 'high_dim':
    #             config.gd_lr = 0.001
    #         if data_kernel == 'high_dim_mixture':
    #             config.gd_lr = 0.001
    #         elif data_kernel == 'imagenet':
    #             config.gd_lr = 0.001
    #     elif config.model_kernel == 'linear':
    #         if data_kernel == 'high_dim':
    #             if num_layers == 1:
    #                 config.gd_lr = 0.001
    #             elif num_layers >= 2:
    #                 config.gd_lr = 0.0001
    #         elif data_kernel == 'imagenet':
    #             if num_layers == 1:
    #                 config.gd_lr = 0.001
    #             elif num_layers >= 2:
    #                 config.gd_lr = 0.0001

    config.examples_per_cat = examples_per_cat  # number of examples per category for imagenet data
    config.holdout = holdout  # whether to use held-out, never before seen categories for evaluation of imagenet
