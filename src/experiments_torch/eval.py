import sys
import os
import random
import pickle
import json
import copy

import matplotlib.pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from transformer import *
from transformer_ff import *
from transformer_moe import *
from data import *
from config import config
from util import *

from IPython.display import display

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("save_path", help="directory to save results to")
parser.add_argument("device_id", type=int, default=0, help="cuda ID to use")
parser.add_argument("model_type", default="interleaved", help="type of transformer to use")
parser.add_argument("num_seeds", type=int, default=5, help="number of seeds")
parser.add_argument("num_layers", type=int, default=1, help="number of layers")

parser.add_argument("data_kernel", help="data type, either high_dim or imagenet")
parser.add_argument("model_kernel", default="softmax", help="kernel used in transformer")

parser.add_argument("categories", type=int, default=25, help="number of categories")
parser.add_argument("dataset_size", type=int, default=125, help="number of in-context examples per contextual dataset")
parser.add_argument("input_size", type=int, default=10, help="dimension of input covariates x_i")
parser.add_argument("embedding_size", type=int, default=5, help="dimension of embedding vectors")
# parser.add_argument("j", type=int, default=5, help="number of alphas to draw for data generation")
parser.add_argument("high_dim_k", type=int, default=5, help="number of clusters for high-dim data")
parser.add_argument("high_dim_lambda", type=float, default=10, help="lambda scaling factor for high-dim data")
parser.add_argument("dist", type=float, default=0.1, help="distance between clusters for high-dim data")
parser.add_argument("mixture_j", type=int, default=10, help="value of J for high-dim mixture data generation")
parser.add_argument("mixture_u_var", type=float, default=5.0, help="variance of u for high-dim mixture data generation")
parser.add_argument("examples_per_cat", type=int, default=10, help="number of examples per category for ImageNet data")

parser.add_argument("num_ff_layers", type=int, default=1, help="number of feedforward layers")
parser.add_argument("ff_hidden_size", type=int, default=20, help="dimension of ff hidden layers")
parser.add_argument("num_heads", type=int, default=1, help="number of non-erasure heads in each attention layer")
parser.add_argument("num_mlps", type=int, default=1, help="number of MLPs for MOE")
parser.add_argument("moe_k", type=int, default=1, help="value of k for MOE")

parser.add_argument("training_data_size", type=int, default=2048, help="number of contextual datasets for training")
parser.add_argument("minibatch_size", type=int, default=512, help="number of contextual datasets per minibatch")
parser.add_argument("training_steps_tf", type=int, default=5000, help="number of training steps for Trained TF")
parser.add_argument("training_steps_gd", type=int, default=5000, help="number of training steps for GD")

parser.add_argument("-o", "--one_hot_emb", action="store_true", default=False,
                    help="use one-hot vectors for embedding matrix W_e")
parser.add_argument("-gd_plus", "--gd_plus", action="store_true", default=False,
                    help="use GD++")
parser.add_argument("-w", "--unique_w_e", action="store_true", default=False,
                    help="use unique w_e for each block of contextual data")
parser.add_argument("-g", "--gd_init", action="store_true", default=False,
                    help="use gd parameters as initial Trained TF params")
parser.add_argument("-v", "--holdout", action="store_true", default=False,
                    help="set true to not use held out dataset for testing in ImageNet data")
parser.add_argument("-f", "--use_mlp", action="store_true", default=False, help="use feedforward network")
parser.add_argument("-s", "--shared_ff", action="store_true", default=False,
                    help="share one feedforward network across all layers")


def create_results_dict(**kwargs):
    return {key: value for key, value in kwargs.items()}


def split_imagenet_data(seed, data, train_classes):
    assert train_classes < data.shape[0]

    gen = torch.Generator().manual_seed(seed)

    indices = torch.randperm(data.shape[0], generator=gen)  # Generate random permutation of indices
    train_indices, test_indices = indices[:train_classes], indices[train_classes:]

    return data[train_indices], data[test_indices]


def eval_loop(model, dataloader):
    model.eval()

    query_loss_avg = 0
    query_acc_avg = 0

    with torch.no_grad():
        for batch, (features, labels) in enumerate(dataloader):
            # Get loss and accuracy
            query_loss, query_acc = model.train_step(features, labels)

            query_loss_avg = query_loss_avg + query_loss
            query_acc_avg = query_acc_avg + query_acc

        query_loss_avg = query_loss_avg / len(dataloader)
        query_acc_avg = query_acc_avg / len(dataloader)

    return query_loss_avg, query_acc_avg


def train_loop(model, optimizer, training_steps, train_dataloader, val_dataloader, eval_dataloader, save_path):
    # Initialize lists for metrics
    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []

    eval_loss = []
    eval_acc = []

    # Initialize values for early stopping
    best_params_trained = None
    best_step = 0
    best_val_loss = torch.inf
    best_val_acc = 0

    # Training loop
    for step in range(training_steps):
        query_loss_train_avg = 0
        query_acc_train_avg = 0

        model.train()
        for batch, (features, labels) in enumerate(train_dataloader):
            # Get train loss and accuracy
            query_loss_train, query_acc_train = model.train_step(features, labels)

            # Backprop and step
            optimizer.zero_grad()
            query_loss_train.backward()
            optimizer.step()

            query_loss_train_avg = query_loss_train_avg + query_loss_train
            query_acc_train_avg = query_acc_train_avg + query_acc_train

        query_loss_train_avg = query_loss_train_avg / len(train_dataloader)
        query_acc_train_avg = query_acc_train_avg / len(train_dataloader)

        # Store loss
        train_loss.append(query_loss_train_avg.item())
        train_acc.append(query_acc_train_avg.item())

        # Evaluate on validation and test data
        model.eval()
        with torch.no_grad():
            query_loss_val, query_acc_val = eval_loop(model, val_dataloader)

            query_loss_eval, query_acc_eval = eval_loop(model, eval_dataloader)

        # Early stopping
        if query_loss_val < best_val_loss:
            best_params_trained = copy.deepcopy(model.state_dict())
            best_step = step
            best_val_loss = query_loss_val
            best_val_acc = query_acc_val
            # gd_best_val_mse = val_prob_dist
            # gd_best_val_top_3_freq = val_top_3_freq

        # Save validation and test metrics
        val_loss.append(query_loss_val.item())
        val_acc.append(query_acc_val.item())

        eval_loss.append(query_loss_eval.item())
        eval_acc.append(query_acc_eval.item())

        # Print progress
        if step % 100 == 0 or step < 5 or step == training_steps - 1:
            print('iter {} | Train Loss: {:.3f} | Train Accuracy: {:.3f} | Val Loss: {:.3f} | '
                  'Val Accuracy: {:.3f} | Eval Loss: {:.3f} | '
                  'Eval Accuracy: {:.3f}'.format(step, query_loss_train_avg.item(), query_acc_train_avg.item(),
                                                 query_loss_val.item(), query_acc_val.item(),
                                                 query_loss_eval.item(), query_acc_eval.item()))

    if save_path is not None:
        print("Saving model...")
        torch.save(model.state_dict(), save_path + '.pth')
        print("Model saved to ", save_path + '.pth')

        print("Saving model with early stopping...")
        torch.save(best_params_trained, save_path + '_es.pth')
        print("Model saved to ", save_path + '_es.pth')

    return train_loss, train_acc, val_loss, val_acc, eval_loss, eval_acc, \
        best_step, best_val_loss, best_val_acc, best_params_trained


def run_experiment(save_path, device_id, model_type, num_seeds=1, num_layers=1, data_kernel='linear',
                   model_kernel='linear',
                   categories=2, dataset_size=20, input_size=2,
                   embedding_size=5, high_dim_k=5, high_dim_lambda=10, dist=0.1,
                   mixture_j=10, mixture_u_var=5,
                   one_hot_emb=False, gd_plus=False,
                   unique_w_e=False, examples_per_cat=10, num_ff_layers=1, ff_hidden_size=10, num_heads=1, num_mlps=1,
                   moe_k=1,
                   shared_ff=False, training_data_size=2048,
                   minibatch_size=512, training_steps_tf=5000, training_steps_gd=5000,
                   holdout=True, gd_init=False,
                   use_mlp=False, seed=42):
    """Run experiments."""
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

    pl.rcParams.update({'font.size': 12})
    pl.rc('axes', labelsize=14)
    pl.rcParams.update({
        "text.usetex": False,
    })

    # initialize experiment parameters
    conf_init(save_path=save_path, model_type=model_type, num_seeds=num_seeds, num_layers=num_layers,
              data_kernel=data_kernel, model_kernel=model_kernel, num_cats=categories,
              c_size=dataset_size, i_size=input_size, e_size=embedding_size,
              high_dim_k=high_dim_k, high_dim_lambda=high_dim_lambda, dist=dist,
              mixture_j=mixture_j, mixture_u_var=mixture_u_var,
              one_hot_emb=one_hot_emb, gd_plus=gd_plus, unique_w_e=unique_w_e,
              examples_per_cat=examples_per_cat,
              num_ff_layers=num_ff_layers, ff_hidden_size=ff_hidden_size,
              num_heads=num_heads, num_mlps=num_mlps, moe_k=moe_k, shared_ff=shared_ff,
              training_data_size=training_data_size, minibatch_size=minibatch_size,
              training_steps_tf=training_steps_tf, training_steps_gd=training_steps_gd,
              holdout=holdout, gd_init=gd_init, use_mlp=use_mlp, num_queries=1)

    # initialize lists to store metrics
    tf_eval_loss_list = []
    tf_eval_acc_list = []
    tf_eval_top_3_freq_list = []
    tf_eval_prob_dist_list = []

    tf_val_loss_list = []
    tf_val_acc_list = []
    tf_val_top_3_freq_list = []
    tf_val_prob_dist_list = []

    tf_train_loss_list = []
    tf_train_acc_list = []
    tf_train_top_3_freq_list = []
    tf_train_prob_dist_list = []

    tf_gd_eval_loss_list = []
    tf_gd_eval_acc_list = []
    tf_gd_eval_top_3_freq_list = []
    tf_gd_eval_prob_dist_list = []

    tf_max_prob_list = []
    tf_target_prob_list = []

    tf_best_step_list = []
    gd_best_step_list = []

    gd_train_loss_list = []
    gd_train_acc_list = []
    gd_train_top_3_freq_list = []
    gd_train_prob_dist_list = []
    gd_train_max_prob_list = []
    gd_train_target_prob_list = []

    gd_eval_loss_list = []
    gd_eval_acc_list = []
    gd_eval_top_3_freq_list = []
    gd_eval_prob_dist_list = []
    gd_eval_max_prob_list = []
    gd_eval_target_prob_list = []

    gd_val_loss_list = []
    gd_val_acc_list = []
    gd_val_top_3_freq_list = []
    gd_val_prob_dist_list = []
    gd_val_max_prob_list = []
    gd_val_target_prob_list = []

    print("dataset size: ", config.dataset_size)
    print("training batch size: ", config.bs_train)
    print("mini batch size: ", config.mb_size)
    print("input size: ", config.input_size)
    print("data generation embedding size: ", config.data_e_size)
    print("params embedding size: ", config.e_size)
    print("gd lr: ", config.gd_lr)
    print("trained tf lr: ", config.lr)
    print("data kernel: ", config.data_kernel)
    print("model kernel: ", config.model_kernel)
    print("num ff layers: ", config.num_ff_layers)
    print("ff hidden layer size: ", config.ff_hidden_size)
    print("number of heads: ", config.num_heads)

    # create save directory
    if not os.path.isdir(config.save_folder):
        os.mkdir(config.save_folder)

    if not os.path.isdir(config.save_folder + "/" + config.model_kernel):
        os.mkdir(config.save_folder + "/" + config.model_kernel)

    # generate W_e's for high-dimensional data
    generator = torch.Generator().manual_seed(seed)
    W_e = torch.randn((config.data_e_size, config.num_cats), generator=generator)
    print(W_e)
    # Generate data
    if config.data_kernel == 'high_dim':
        train_data = HighDimCategorical(device=device, model_type=config.model_type, seed=100, batches=config.bs_train,
                                        i_size=config.input_size, c_size=config.dataset_size,
                                        data_e_size=config.data_e_size, params_e_size=config.e_size,
                                        cats=config.num_cats, k=config.high_dim_k,
                                        dist=config.dist, l=config.high_dim_lambda, W_e=W_e, input_range=1)
        val_data = HighDimCategorical(device=device, model_type=config.model_type, seed=100 + config.bs_train,
                                      batches=config.bs_eval,
                                      i_size=config.input_size, c_size=config.dataset_size,
                                      data_e_size=config.data_e_size, params_e_size=config.e_size,
                                      cats=config.num_cats, k=config.high_dim_k,
                                      dist=config.dist, l=config.high_dim_lambda, W_e=W_e, input_range=1)
        eval_data = HighDimCategorical(device=device, model_type=config.model_type,
                                       seed=100 + config.bs_train + config.bs_eval,
                                       batches=config.bs_eval,
                                       i_size=config.input_size, c_size=config.dataset_size,
                                       data_e_size=config.data_e_size, params_e_size=config.e_size,
                                       cats=config.num_cats, k=config.high_dim_k,
                                       dist=config.dist, l=config.high_dim_lambda, W_e=W_e, input_range=1)
    if config.data_kernel == 'high_dim_mixture':
        train_data = HighDimMixtureCategorical(device=device, model_type=config.model_type, seed=100,
                                               batches=config.bs_train,
                                               i_size=config.input_size, c_size=config.dataset_size,
                                               data_e_size=config.data_e_size, params_e_size=config.e_size,
                                               cats=config.num_cats, k=config.high_dim_k,
                                               dist=config.dist, l=config.high_dim_lambda, W_e=W_e,
                                               J=mixture_j, u_var=mixture_u_var, input_range=1)
        val_data = HighDimMixtureCategorical(device=device, model_type=config.model_type, seed=100 + config.bs_train,
                                             batches=config.bs_eval,
                                             i_size=config.input_size, c_size=config.dataset_size,
                                             data_e_size=config.data_e_size, params_e_size=config.e_size,
                                             cats=config.num_cats, k=config.high_dim_k,
                                             dist=config.dist, l=config.high_dim_lambda, W_e=W_e,
                                             J=mixture_j, u_var=mixture_u_var, input_range=1)
        eval_data = HighDimMixtureCategorical(device=device, model_type=config.model_type,
                                              seed=100 + config.bs_train + config.bs_eval,
                                              batches=config.bs_eval,
                                              i_size=config.input_size, c_size=config.dataset_size,
                                              data_e_size=config.data_e_size, params_e_size=config.e_size,
                                              cats=config.num_cats, k=config.high_dim_k,
                                              dist=config.dist, l=config.high_dim_lambda, W_e=W_e,
                                              J=mixture_j, u_var=mixture_u_var, input_range=1)
    elif config.data_kernel == 'imagenet':
        data = torch.from_numpy(np.load("train_features.npy"))

        if config.holdout:
            imagenet_train_data, imagenet_test_data = split_imagenet_data(seed=11, data=data, train_classes=900)
        else:
            imagenet_train_data = data
            imagenet_test_data = torch.from_numpy(np.load("val_features.npy"))

        # with held out data, use the training classes for both train and val datasets, so
        # that eval is truly never seen before
        train_data = ImageNet(device=device, model_type=config.model_type, seed=100, batches=config.bs_train,
                              data=imagenet_train_data, params_e_size=config.e_size, cats=config.num_cats,
                              examples_per_cat=config.examples_per_cat)
        val_data = ImageNet(device=device, model_type=config.model_type, seed=100 + config.bs_train,
                            batches=config.bs_eval,
                            data=imagenet_train_data, params_e_size=config.e_size, cats=config.num_cats,
                            examples_per_cat=config.examples_per_cat)
        eval_data = ImageNet(device=device, model_type=config.model_type, seed=100 + config.bs_train + config.bs_eval,
                             batches=config.bs_eval,
                             data=imagenet_test_data, params_e_size=config.e_size, cats=config.num_cats,
                             examples_per_cat=config.examples_per_cat)
    elif config.data_kernel == 'random_grid':
        train_data = RandomGridData(device=device, model_type=config.model_type, seed=100, batches=config.bs_train,
                                    i_size=config.input_size, c_size=config.dataset_size,
                                    data_e_size=config.data_e_size, params_e_size=config.e_size,
                                    cats=config.num_cats, W_e=W_e, input_range=1)
        val_data = RandomGridData(device=device, model_type=config.model_type, seed=100 + config.bs_train,
                                  batches=config.bs_eval,
                                  i_size=config.input_size, c_size=config.dataset_size,
                                  data_e_size=config.data_e_size, params_e_size=config.e_size,
                                  cats=config.num_cats, W_e=W_e, input_range=1)
        eval_data = RandomGridData(device=device, model_type=config.model_type,
                                   seed=100 + config.bs_train + config.bs_eval,
                                   batches=config.bs_eval,
                                   i_size=config.input_size, c_size=config.dataset_size,
                                   data_e_size=config.data_e_size, params_e_size=config.e_size,
                                   cats=config.num_cats, W_e=W_e, input_range=1)

    train_dataloader = DataLoader(train_data, batch_size=config.mb_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=config.mb_size, shuffle=False)
    eval_dataloader = DataLoader(eval_data, batch_size=config.mb_size, shuffle=False)

    # begin training
    for cur_seed in range(config.num_seeds):
        print("Current seed: ", cur_seed)
        config.seed = cur_seed

        # Initialize GD model
        if config.model_type == 'interleaved':
            gd_model = TransformerGDInterleaved(num_layers=config.num_layers,
                                                input_size=config.input_size,
                                                num_categories=config.num_cats,
                                                emb_size=config.e_size,
                                                init_scale=config.init_scale,
                                                include_query=False,
                                                kernel=config.model_kernel,
                                                num_queries=1,
                                                use_mlp=config.use_mlp,
                                                device=device,
                                                init_seed=cur_seed).to(device)
        elif config.model_type == 'linear_approx':
            gd_model = TransformerGDLinearApprox(num_layers=config.num_layers,
                                                 input_size=config.input_size,
                                                 num_categories=config.num_cats,
                                                 emb_size=config.e_size,
                                                 init_scale=config.init_scale,
                                                 include_query=False,
                                                 kernel=config.model_kernel,
                                                 num_queries=1,
                                                 use_mlp=config.use_mlp,
                                                 device=device,
                                                 init_seed=cur_seed).to(device)
        elif config.model_type == 'feedforward':
            gd_model = TransformerGDFeedforward(num_layers=config.num_layers,
                                                num_ff_layers=config.num_ff_layers,
                                                num_heads=config.num_heads,
                                                input_size=config.input_size,
                                                num_categories=config.num_cats,
                                                emb_size=config.e_size,
                                                ff_hidden_size=config.ff_hidden_size,
                                                init_scale=config.init_scale,
                                                include_query=False,
                                                kernel=config.model_kernel,
                                                num_queries=1,
                                                shared_ff=shared_ff,
                                                device=device,
                                                init_seed=cur_seed).to(device)
        elif config.model_type == 'moe':
            gd_model = TransformerGDMOE(num_layers=config.num_layers,
                                        num_ff_layers=config.num_ff_layers,
                                        num_heads=config.num_heads,
                                        num_mlps=config.num_mlps,
                                        input_size=config.input_size,
                                        num_categories=config.num_cats,
                                        emb_size=config.e_size,
                                        ff_hidden_size=config.ff_hidden_size,
                                        k=config.moe_k,
                                        init_scale=config.init_scale,
                                        include_query=False,
                                        kernel=config.model_kernel,
                                        num_queries=1,
                                        shared_ff=shared_ff,
                                        device=device,
                                        init_seed=cur_seed).to(device)

        for name, param in gd_model.named_parameters():
            if param.requires_grad:
                print(f"Trainable Parameter: {name}, Shape: {param.shape}")

        # Define GD optimizer
        gd_optimizer = torch.optim.Adam(gd_model.parameters(), lr=config.gd_lr)

        # Train GD model
        print("Starting GD training...")

        gd_train_loss, gd_train_acc, gd_val_loss, gd_val_acc, gd_train_eval_loss, gd_train_eval_acc, gd_best_step, \
            gd_best_val_loss, gd_best_val_acc, gd_best_params_trained = train_loop(gd_model,
                                                                                   gd_optimizer,
                                                                                   config.training_steps_gd,
                                                                                   train_dataloader,
                                                                                   val_dataloader,
                                                                                   eval_dataloader,
                                                                                   '{}/{}/gd_seed_{}'.format(save_path,
                                                                                                             config.model_kernel,
                                                                                                             cur_seed))

        print("GD training complete")
        print("Best Step: ", gd_best_step)

        # store GD training and validation metrics
        gd_train_loss_list.append(gd_train_loss)
        gd_train_acc_list.append(gd_train_acc)
        # gd_train_eval_top_3_freq_list.append(gd_train_top_3_freq)
        # gd_train_eval_prob_dist_list.append(gd_train_prob_dist)
        # gd_train_eval_max_prob_list.append(gd_train_max_prob)
        # gd_train_eval_target_prob_list.append(gd_train_target_prob)

        gd_val_loss_list.append(gd_val_loss)
        gd_val_acc_list.append(gd_val_acc)
        # gd_val_top_3_freq_list.append(gd_val_top_3_freq)
        # gd_val_prob_dist_list.append(gd_val_prob_dist)
        # gd_val_max_prob_list.append(gd_val_mean_max_prob)
        # gd_val_target_prob_list.append(gd_val_mean_target_prob)

        gd_eval_loss_list.append(gd_train_eval_loss)
        gd_eval_acc_list.append(gd_train_eval_acc)

        gd_best_step_list.append(gd_best_step)

        # evaluate gd and gd++ performance on evaluation data
        loss_gd, acc_gd = eval_loop(gd_model, eval_dataloader)

        tf_gd_eval_loss = []
        tf_gd_eval_acc = []

        for i in range(config.training_steps_tf):
            tf_gd_eval_loss.append(loss_gd.item())
            tf_gd_eval_acc.append(acc_gd.item())

        tf_gd_eval_loss_list.append(tf_gd_eval_loss)
        tf_gd_eval_acc_list.append(tf_gd_eval_acc)

        # Initialize TF model
        if config.model_type == 'interleaved':
            tf_model = TransformerInterleaved(num_layers=config.num_layers,
                                              input_size=config.input_size,
                                              num_categories=config.num_cats,
                                              emb_size=config.e_size,
                                              init_scale=config.init_scale,
                                              include_query=False,
                                              kernel=config.model_kernel,
                                              num_queries=1,
                                              use_mlp=config.use_mlp,
                                              device=device,
                                              init_seed=cur_seed).to(device)
        elif config.model_type == 'linear_approx':
            tf_model = TransformerLinearApprox(num_layers=config.num_layers,
                                               input_size=config.input_size,
                                               num_categories=config.num_cats,
                                               emb_size=config.e_size,
                                               init_scale=config.init_scale,
                                               include_query=False,
                                               kernel=config.model_kernel,
                                               num_queries=1,
                                               use_mlp=config.use_mlp,
                                               device=device,
                                               init_seed=cur_seed).to(device)
        elif config.model_type == 'feedforward':
            tf_model = TransformerFeedforward(num_layers=config.num_layers,
                                              num_ff_layers=config.num_ff_layers,
                                              num_heads=config.num_heads,
                                              input_size=config.input_size,
                                              num_categories=config.num_cats,
                                              emb_size=config.e_size,
                                              ff_hidden_size=config.ff_hidden_size,
                                              init_scale=config.init_scale,
                                              include_query=False,
                                              kernel=config.model_kernel,
                                              num_queries=1,
                                              shared_ff=shared_ff,
                                              device=device,
                                              init_seed=cur_seed).to(device)
        elif config.model_type == 'moe':
            tf_model = TransformerMOE(num_layers=config.num_layers,
                                      num_ff_layers=config.num_ff_layers,
                                      num_heads=config.num_heads,
                                      num_mlps=config.num_mlps,
                                      input_size=config.input_size,
                                      num_categories=config.num_cats,
                                      emb_size=config.e_size,
                                      ff_hidden_size=config.ff_hidden_size,
                                      k=config.moe_k,
                                      init_scale=config.init_scale,
                                      include_query=False,
                                      kernel=config.model_kernel,
                                      num_queries=1,
                                      shared_ff=shared_ff,
                                      device=device,
                                      init_seed=cur_seed).to(device)

        for name, param in tf_model.named_parameters():
            if param.requires_grad:
                print(f"Trainable Parameter: {name}, Shape: {param.shape}")

        # Define TF optimizer
        tf_optimizer = torch.optim.AdamW(tf_model.parameters(), lr=config.gd_lr, betas=[config.b1, config.b2],
                                         weight_decay=config.wd)

        print("Starting Trained TF training...")

        tf_train_loss, tf_train_acc, tf_val_loss, tf_val_acc, tf_eval_loss, tf_eval_acc, tf_best_step, \
            tf_best_val_loss, tf_best_val_acc, tf_best_params_trained = train_loop(tf_model,
                                                                                   tf_optimizer,
                                                                                   config.training_steps_tf,
                                                                                   train_dataloader,
                                                                                   val_dataloader,
                                                                                   eval_dataloader,
                                                                                   '{}/{}/tf_seed_{}'.format(save_path,
                                                                                                             config.model_kernel,
                                                                                                             cur_seed))

        print("Trained TF training complete")
        print("Best Step: ", tf_best_step)

        # store TF training and validation metrics
        tf_train_loss_list.append(tf_train_loss)
        tf_train_acc_list.append(tf_train_acc)
        # gd_train_eval_top_3_freq_list.append(gd_train_top_3_freq)
        # gd_train_eval_prob_dist_list.append(gd_train_prob_dist)
        # gd_train_eval_max_prob_list.append(gd_train_max_prob)
        # gd_train_eval_target_prob_list.append(gd_train_target_prob)

        tf_val_loss_list.append(tf_val_loss)
        tf_val_acc_list.append(tf_val_acc)
        # gd_val_top_3_freq_list.append(gd_val_top_3_freq)
        # gd_val_prob_dist_list.append(gd_val_prob_dist)
        # gd_val_max_prob_list.append(gd_val_mean_max_prob)
        # gd_val_target_prob_list.append(gd_val_mean_target_prob)

        tf_eval_loss_list.append(tf_eval_loss)
        tf_eval_acc_list.append(tf_eval_acc)

        tf_best_step_list.append(tf_best_step)

        # save results into JSON
        try:
            results = create_results_dict(tf_eval_loss_list=tf_eval_loss_list,
                                          tf_eval_acc_list=tf_eval_acc_list,
                                          tf_eval_top_3_freq_list=tf_eval_top_3_freq_list,
                                          tf_eval_prob_dist_list=tf_eval_prob_dist_list,
                                          tf_val_loss_list=tf_val_loss_list,
                                          tf_val_acc_list=tf_val_acc_list,
                                          tf_val_top_3_freq_list=tf_val_top_3_freq_list,
                                          tf_val_prob_dist_list=tf_val_prob_dist_list,
                                          tf_train_loss_list=tf_train_loss_list,
                                          tf_train_acc_list=tf_train_acc_list,
                                          tf_train_top_3_freq_list=tf_train_top_3_freq_list,
                                          tf_train_prob_dist_list=tf_train_prob_dist_list,
                                          tf_max_prob_list=tf_max_prob_list,
                                          tf_target_prob_list=tf_target_prob_list,
                                          tf_best_step_list=tf_best_step_list,
                                          tf_gd_eval_loss_list=tf_gd_eval_loss_list,
                                          tf_gd_eval_acc_list=tf_gd_eval_acc_list,
                                          tf_gd_eval_top_3_freq_list=tf_gd_eval_top_3_freq_list,
                                          tf_gd_eval_prob_dist_list=tf_gd_eval_prob_dist_list,
                                          gd_train_loss_list=gd_train_loss_list,
                                          gd_train_acc_list=gd_train_acc_list,
                                          gd_train_top_3_freq_list=gd_train_top_3_freq_list,
                                          gd_train_prob_dist_list=gd_train_prob_dist_list,
                                          gd_val_loss_list=gd_val_loss_list,
                                          gd_val_acc_list=gd_val_acc_list,
                                          gd_val_top_3_freq_list=gd_val_top_3_freq_list,
                                          gd_val_prob_dist_list=gd_val_prob_dist_list,
                                          gd_eval_loss_list=gd_eval_loss_list,
                                          gd_eval_acc_list=gd_eval_acc_list,
                                          gd_eval_top_3_freq_list=gd_eval_top_3_freq_list,
                                          gd_eval_prob_dist_list=gd_eval_prob_dist_list,
                                          gd_best_step_list=gd_best_step_list)

            with open(config.save_folder + "/" + config.model_kernel + "/results.json", "w") as fp:
                json.dump(results, fp)

            print("Done writing results into json file")
        except Exception as e:
            print("Error getting results")
            print(e)

    # don't save imagenet data
    config.train_features = None
    config.val_features = None
    config.train_lens = None
    config.val_lens = None

    # save config into JSON
    with open(config.save_folder + "/" + config.model_kernel + "/config.json", "w") as fp:
        json.dump(config.to_dict(), fp)

    print("Done saving config into json file")


if __name__ == '__main__':
    # Checking the number of devices we have here
    num_devices = torch.cuda.device_count()
    print(f"Number of available devices: {num_devices}")

    if torch.cuda.is_available():
        for i in range(num_devices):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {gpu_name}")

    seed = 10  # Choose any integer seed

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single-GPU use
    torch.cuda.manual_seed_all(seed)  # For multi-GPU use

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parser.parse_args()

    assert 0 <= args.device_id < num_devices

    run_experiment(args.save_path,
                   args.device_id,
                   args.model_type,
                   args.num_seeds,
                   args.num_layers,
                   args.data_kernel,
                   args.model_kernel,
                   args.categories,
                   args.dataset_size,
                   args.input_size,
                   args.embedding_size,
                   args.high_dim_k,
                   args.high_dim_lambda,
                   args.dist,
                   args.mixture_j,
                   args.mixture_u_var,
                   args.one_hot_emb,
                   args.gd_plus,
                   args.unique_w_e,
                   args.examples_per_cat,
                   args.num_ff_layers,
                   args.ff_hidden_size,
                   args.num_heads,
                   args.num_mlps,
                   args.moe_k,
                   args.shared_ff,
                   args.training_data_size,
                   args.minibatch_size,
                   args.training_steps_tf,
                   args.training_steps_gd,
                   args.holdout,
                   args.gd_init,
                   args.use_mlp,
                   seed)
