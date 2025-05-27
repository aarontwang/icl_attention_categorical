import sys
import os

import matplotlib.pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

import pickle
import json

from transformer import Transformer
from data import create_cat_data_grid, create_cat_data_random_grid, create_cat_data_high_dim, \
    create_img_data, create_weights, create_w_e
from config import config
from train import *
from eval import *
from util import *
import plot

from IPython.display import display

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("save_path", help="directory to save results to")
parser.add_argument("num_seeds", type=int, help="number of seeds")
parser.add_argument("num_layers", type=int, help="number of layers")
parser.add_argument("data_kernel", help="kernel used to generate data")
parser.add_argument("model_kernel", help="kernel used in transformer")
parser.add_argument("categories", type=int, help="number of categories")
parser.add_argument("gamma", type=float, help="parameter for exp kernel")
parser.add_argument("sigma", type=float, help="parameter for rbf kernel")
parser.add_argument("dataset_size", type=int, help="number of in-context examples")
parser.add_argument("input_size", type=int, help="dimension of input vectors")
parser.add_argument("embedding_size", type=int, help="dimension of embedding vectors")
parser.add_argument("j", type=int, help="number of alphas to draw for data generation")
parser.add_argument("k", type=int, help="number of clusters for high-dim data")
parser.add_argument("l", type=float, help="lambda scaling factor for high-dim data")
parser.add_argument("dist", type=float, help="distance between clusters for high-dim data")
parser.add_argument("num_heads_gd", type=int, help="number of heads in GD transformer")
parser.add_argument("num_heads_tr", type=int, help="number of heads in trained TF")
parser.add_argument("examples_per_cat", type=int, help="number of examples per category for ImageNet data")
parser.add_argument("training_data_size", type=int, help="number of contextual datasets for training")
parser.add_argument("minibatch_size", type=int, help="number of contextual datasets per minibatch")
parser.add_argument("training_steps_tf", type=int, help="number of training steps for Trained TF")
parser.add_argument("training_steps_gd", type=int, help="number of training steps for GD")


parser.add_argument("-r", "--recurrent", action="store_false",
                    help="use recurrent layers")
parser.add_argument("-bh", "--bias_head", action="store_true",
                    help="use bias head")
parser.add_argument("-bd", "--bias_data", action="store_true",
                    help="use biased data")
parser.add_argument("-o", "--one_hot_emb", action="store_true",
                    help="use one-hot vectors for embedding matrix W_e")
parser.add_argument("-reg", "--regularize", action="store_true",
                    help="regularize kernel")
parser.add_argument("-gd_plus", "--gd_plus", action="store_true",
                    help="use GD++")
parser.add_argument("-d", "--diag", action="store_true",
                    help="use diagonal matrix for GD++")
parser.add_argument("-es", "--early_stopping", action="store_true",
                    help="use early stopping")
parser.add_argument("-w", "--unique_w_e", action="store_true",
                    help="use unique w_e for each block of contextual data")
parser.add_argument("-g", "--gd_init", action="store_true",
                    help="use gd parameters as initial Trained TF params")
parser.add_argument("-v", "--holdout", action="store_true",
                    help="use held out dataset for testing in ImageNet data")
parser.add_argument("-m", "--linear_approx", action="store_true", help="use linear approximation")
parser.add_argument("-f", "--use_mlp", action="store_true", help="use feedforward network")

def get_results_dict(loss_trans_list, loss_trans_train_list, acc_trans_list, top_3_freq_trans_list,
                     prob_dist_trans_list, best_idx_trans_list,
                     losses_gd_list, acc_gd_list, top_3_freq_gd_list, prob_dist_gd_list,
                     max_prob_list, target_prob_list, cos_sim_list, grad_norm_list, p_norm_list,
                     gd_train_loss_list, gd_train_acc_list, gd_train_top_3_freq_list, gd_train_prob_dist_list,
                     gd_train_max_prob_list, gd_train_target_prob_list,
                     gd_val_loss_list, gd_val_acc_list, gd_val_top_3_freq_list, gd_val_prob_dist_list,
                     gd_val_max_prob_list, gd_val_target_prob_list, gd_val_best_steps
                     ):
    results = {'loss_trans_list': [[y.tolist() for y in x if y is not None] for x in loss_trans_list],
               'loss_trans_train_list': loss_trans_train_list,
               'acc_trans_list': [[y.tolist() for y in x if y is not None] for x in acc_trans_list],
               'top_3_freq_trans_list': [[y.tolist() for y in x if y is not None] for x in top_3_freq_trans_list],
               'prob_dist_trans_list': [[y.tolist() for y in x if y is not None] for x in prob_dist_trans_list],
               'best_idx_trans_list': best_idx_trans_list,
               'losses_gd_list': [[y.tolist() for y in x if y is not None] for x in losses_gd_list],
               'acc_gd_list': [[y.tolist() for y in x if y is not None] for x in acc_gd_list],
               'top_3_freq_gd_list': [[y.tolist() for y in x if y is not None] for x in top_3_freq_gd_list],
               'prob_dist_gd_list': [[y.tolist() for y in x if y is not None] for x in prob_dist_gd_list],
               'max_prob_list': [[y.tolist() for y in x if y is not None] for x in max_prob_list],
               'target_prob_list': [[y.tolist() for y in x if y is not None] for x in target_prob_list],
               'cos_sim_list': [[y.tolist() for y in x if y is not None] for x in cos_sim_list],
               'grad_norm_list': [[y.tolist() for y in x if y is not None] for x in grad_norm_list],
               'p_norm_list': [[y.tolist() for y in x if y is not None] for x in p_norm_list],
               'gd_train_loss_list': [[y.tolist() for y in x if y is not None] for x in gd_train_loss_list],
               'gd_train_acc_list': [[y.tolist() for y in x if y is not None] for x in gd_train_acc_list],
               'gd_train_top_3_freq_list': [[y.tolist() for y in x if y is not None] for x in gd_train_top_3_freq_list],
               'gd_train_prob_dist_list': [[y.tolist() for y in x if y is not None] for x in gd_train_prob_dist_list],
               'gd_train_max_prob_list': [[y.tolist() for y in x if y is not None] for x in gd_train_max_prob_list],
               'gd_train_target_prob_list': [[y.tolist() for y in x if y is not None] for x in
                                             gd_train_target_prob_list],
               'gd_val_loss_list': [[y.tolist() for y in x if y is not None] for x in gd_val_loss_list],
               'gd_val_acc_list': [[y.tolist() for y in x if y is not None] for x in gd_val_acc_list],
               'gd_val_top_3_freq_list': [[y.tolist() for y in x if y is not None] for x in gd_val_top_3_freq_list],
               'gd_val_prob_dist_list': [[y.tolist() for y in x if y is not None] for x in gd_val_prob_dist_list],
               'gd_val_max_prob_list': [[y.tolist() for y in x if y is not None] for x in gd_val_max_prob_list],
               'gd_val_target_prob_list': [[y.tolist() for y in x if y is not None] for x in gd_val_target_prob_list],
               'gd_val_best_step': [x for x in gd_val_best_steps]
               }

    return results


def get_data(data_creator, rng, kernel, batch_size, input_size, dataset_size, embedding_size, cats, j, sigma, gamma,
             input_range, weight_scale, bias_data, k, dist, l, w_e, data, data_len, examples_per_cat, min_len):
    """Loads data."""
    if kernel == 'grid':
        data = data_creator(jax.random.split(rng, num=batch_size),
                            input_size,
                            dataset_size,
                            embedding_size,
                            cats,
                            input_range,
                            weight_scale,
                            bias_data)
        data = grid_to_reg(data)
    elif kernel == 'random_grid':
        data = data_creator(jax.random.split(rng, num=batch_size),
                            input_size,
                            dataset_size,
                            embedding_size,
                            cats,
                            input_range,
                            weight_scale,
                            w_e,
                            bias_data)
    elif kernel == 'high_dim':
        sharding = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))

        data = data_creator(jax.device_put(jax.random.split(rng, num=batch_size), sharding.reshape((2, 1))),
                            input_size,
                            dataset_size,
                            embedding_size,
                            cats,
                            k,
                            dist,
                            l,
                            w_e,
                            input_range)

        data = list(data)
        data[0] = jax.device_put(data[0], sharding.reshape(2, 1, 1))
        data = tuple(data)
    elif kernel == 'imagenet':
        sharding = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))

        data = data_creator(jax.device_put(jax.random.split(rng, num=batch_size), sharding.reshape((2, 1))),
                            data,
                            data_len,
                            embedding_size,
                            cats,
                            examples_per_cat,
                            min_len)

        data = list(data)
        data[0] = jax.device_put(data[0], sharding.reshape(2, 1, 1))
        data = tuple(data)

    return data


def run_experiment(save_path, num_seeds=1, num_layers=1, recurrent=True, data_kernel='linear', modeL_kernel='linear',
                   categories=2, use_bias_head=False, use_bias_data=False,
                   gamma=None, sigma=None, dataset_size=20, input_size=2, embedding_size=5, j=5, k=5, l=3, dist=0.1,
                   one_hot_emb=False, num_heads_gd=1, num_heads_tr=1, gd_plus=False, diag=False,
                   early_stopping=False, unique_w_e=False, examples_per_cat=10, training_data_size=2048,
                   minibatch_size=512, training_steps_tf=5000, training_steps_gd=5000, holdout=False, gd_init=False,
                   linear_approx=False, use_mlp=False):
    """Run experiments."""
    pl.rcParams.update({'font.size': 12})
    pl.rc('axes', labelsize=14)
    pl.rcParams.update({
        "text.usetex": False,
    })

    # initialize experiment parameters
    conf_init(save_path, num_seeds, num_layers, recurrent, data_kernel, modeL_kernel, categories, use_bias_head,
              use_bias_data, gamma, sigma, dataset_size, input_size, embedding_size, j, k, l, dist, one_hot_emb,
              num_heads_gd, num_heads_tr, gd_plus, diag, early_stopping, unique_w_e, examples_per_cat,
              training_data_size, minibatch_size, training_steps_tf, training_steps_gd,
              hold_out=holdout, gd_init=gd_init, linear_approx=linear_approx, use_mlp=use_mlp, num_queries=1)

    # set dataloader
    if config.data_kernel == 'grid':
        data_creator = vmap(create_cat_data_grid,
                            in_axes=(0, None, None, None, None, None, None, None), out_axes=0)
    elif config.data_kernel == 'random_grid':
        data_creator = vmap(create_cat_data_random_grid,
                            in_axes=(0, None, None, None, None, None, None, None, None), out_axes=0)
    elif config.data_kernel == 'high_dim':
        data_creator = vmap(create_cat_data_high_dim,
                            in_axes=(0, None, None, None, None, None, None, None, None, None), out_axes=0)
    elif config.data_kernel == 'imagenet':
        data_creator = vmap(create_img_data,
                    in_axes=(0, None, None, None, None, None, None), out_axes=0)

    change_dataloader()

    # initialize empty lists to store metrics
    loss_trans_list = [[] for _ in range(config.num_seeds)]
    acc_trans_list = [[] for _ in range(config.num_seeds)]
    top_3_freq_trans_list = [[] for _ in range(config.num_seeds)]
    prob_dist_trans_list = [[] for _ in range(config.num_seeds)]

    loss_trans_train_list = [[] for _ in range(config.num_seeds)]

    losses_gd_list = [[] for _ in range(config.num_seeds)]
    acc_gd_list = [[] for _ in range(config.num_seeds)]
    top_3_freq_gd_list = [[] for _ in range(config.num_seeds)]
    prob_dist_gd_list = [[] for _ in range(config.num_seeds)]

    max_prob_list = [[] for _ in range(config.num_seeds)]
    target_prob_list = [[] for _ in range(config.num_seeds)]

    val_losses = [[] for _ in range(config.num_seeds)]
    val_acc = [[] for _ in range(config.num_seeds)]
    val_top_3_freq = [[] for _ in range(config.num_seeds)]
    val_prob_dist = [[] for _ in range(config.num_seeds)]
    val_mean_max_prob_list = [[] for _ in range(config.num_seeds)]
    val_mean_target_prob_list = [[] for _ in range(config.num_seeds)]
    best_idx_trans_list = []

    gd_train_eval_loss_list = []
    gd_train_eval_acc_list = []
    gd_train_eval_top_3_freq_list = []
    gd_train_eval_prob_dist_list = []
    gd_train_eval_max_prob_list = []
    gd_train_eval_target_prob_list = []

    gd_val_loss_list = []
    gd_val_acc_list = []
    gd_val_top_3_freq_list = []
    gd_val_prob_dist_list = []
    gd_val_max_prob_list = []
    gd_val_target_prob_list = []
    gd_val_best_steps = []

    cos_sim_list = [[] for _ in range(config.num_seeds)]
    grad_norm_list = [[] for _ in range(config.num_seeds)]
    p_norm_list = [[] for _ in range(config.num_seeds)]

    print("gamma: ", config.gamma)
    print("sigma: ", config.sigma)
    print("dataset size: ", config.dataset_size)
    print("training batch size: ", config.bs_train)
    print("mini batch size: ", config.mb_size)
    print("lambda: ", config.l)
    print("k: ", config.k)
    print("input size: ", config.input_size)
    print("data embedding size: ", config.data_e_size)
    print("params embedding size: ", config.e_size)
    print("gd lr: ", config.gd_lr)
    print("trained tf lr: ", config.lr)
    print("data kernel: ", config.data_kernel)
    print("model kernel: ", config.model_kernel)
    print("GD training steps: ", config.training_steps_gd_constructed)

    # create save directory
    if not os.path.isdir(config.save_folder):
        os.mkdir(config.save_folder)

    if not os.path.isdir(config.save_folder + "/" + config.model_kernel):
        os.mkdir(config.save_folder + "/" + config.model_kernel)

    # generate evaluation data
    eval_rng = jax.random.PRNGKey(5)
    eval_rng, val_rng, train_data_rng, w_e_rng = jax.random.split(eval_rng, num=4)

    W_e = create_w_e(w_e_rng, config.data_e_size, config.cats)
    w_e_percentage = 100 * jnp.abs(jnp.expand_dims(jnp.mean(W_e, axis=1), axis=-1) / W_e)

    w_e_data = {'W_e': [x.tolist() for x in W_e],
                'W_e_mean': jnp.mean(W_e, axis=1).tolist(),
                'W_e_percentage': [x.tolist() for x in w_e_percentage]}

    with open(config.save_folder + "/" + config.model_kernel + "/w_e_data.json", "w") as fp:
        json.dump(w_e_data, fp)

    # begin experiments
    for cur_seed in range(config.num_seeds):
        print("Current seed: ", cur_seed)
        config.seed = cur_seed
        optimizer, train_state, _, rng = init(False)
        rng, data_rng = jax.random.split(rng, num=2)

        if config.analyze:
            lr_min = 1

            # create gd weights
            params_gd = create_weights(config.input_size, config.e_size, config.dataset_size, config.cats,
                                       lr_min, jnp.array([lr_min]), gd_deq=config.gd_deq,
                                       num_layers=config.num_layers, use_bias_head=config.bias_head,
                                       gd_plus=config.gd_plus, widening_factor=config.widening_factor,
                                       one_hot_emb=config.one_hot_emb,
                                       linear_approx=config.linear_approx, use_mlp=config.use_mlp, rng=rng)

            # print("before training:\n")
            # for param in params_gd:
            #     with jnp.printoptions(threshold=sys.maxsize):
            #         print(param)
            #         print(params_gd[param]['w'])
            print("Starting GD training...")

            params_gd, data_rng, gd_val_loss, gd_val_acc, gd_val_top_3_freq, gd_val_prob_dist, \
                gd_val_mean_max_prob, gd_val_mean_target_prob, gd_train_loss, gd_train_acc, gd_train_top_3_freq, \
                gd_train_prob_dist, gd_train_max_prob, gd_train_target_prob, gd_val_best_step = \
                pre_train_gd_classification(train_data_rng, val_rng, eval_rng, None if config.unique_w_e else W_e,
                                            params_gd, linear=(config.model_kernel == 'linear'))

            print("GD training complete")
            print("Best Step: ", gd_val_best_step)
            # print("after training:\n")
            # for param in params_gd:
            #     with jnp.printoptions(threshold=sys.maxsize):
            #         print(param)
            #         print(params_gd[param]['w'])

            # store GD training and validation metrics
            gd_train_eval_loss_list.append(gd_train_loss)
            gd_train_eval_acc_list.append(gd_train_acc)
            gd_train_eval_top_3_freq_list.append(gd_train_top_3_freq)
            gd_train_eval_prob_dist_list.append(gd_train_prob_dist)
            gd_train_eval_max_prob_list.append(gd_train_max_prob)
            gd_train_eval_target_prob_list.append(gd_train_target_prob)

            gd_val_loss_list.append(gd_val_loss)
            gd_val_acc_list.append(gd_val_acc)
            gd_val_top_3_freq_list.append(gd_val_top_3_freq)
            gd_val_prob_dist_list.append(gd_val_prob_dist)
            gd_val_max_prob_list.append(gd_val_mean_max_prob)
            gd_val_target_prob_list.append(gd_val_mean_target_prob)

            gd_val_best_steps.append(gd_val_best_step)

            # save GD parameters
            with open('{}/{}/params_gd_seed_{}.pickle'.format(config.save_folder, config.model_kernel,
                                                              cur_seed), 'wb') as handle:
                pickle.dump(params_gd, handle, protocol=5)

        imagenet = config.data_kernel == 'imagenet'

        # generate validation data
        val_data = get_data(data_creator, rng=val_rng, kernel=config.data_kernel,
                            batch_size=config.bs_eval, input_size=config.input_size,
                            dataset_size=config.dataset_size,
                            embedding_size=config.e_size, cats=config.cats, j=config.j,
                            sigma=config.sigma, gamma=config.gamma, input_range=config.input_range,
                            weight_scale=config.weight_scale, bias_data=config.bias_data,
                            k=config.k, dist=config.dist, l=config.l, w_e=None if config.unique_w_e else W_e,
                            data=config.train_features, data_len=config.train_lens,
                            examples_per_cat=config.examples_per_cat, min_len=config.train_min_len)

        # generate test data
        eval_data = get_data(data_creator, rng=eval_rng, kernel=config.data_kernel,
                             batch_size=config.bs_eval, input_size=config.input_size,
                             dataset_size=config.dataset_size,
                             embedding_size=config.e_size, cats=config.cats, j=config.j,
                             sigma=config.sigma, gamma=config.gamma, input_range=config.input_range,
                             weight_scale=config.weight_scale, bias_data=config.bias_data,
                             k=config.k, dist=config.dist, l=config.l, w_e=None if config.unique_w_e else W_e,
                             data=config.val_features, data_len=config.val_lens,
                             examples_per_cat=config.examples_per_cat, min_len=config.val_min_len
                             )

        # generate training data
        # train_data = get_data(data_creator, rng=train_data_rng, kernel=config.data_kernel,
        #                       batch_size=config.bs_train, input_size=config.input_size,
        #                       dataset_size=config.dataset_size,
        #                       embedding_size=config.e_size, cats=config.cats, j=config.j,
        #                       sigma=config.sigma, gamma=config.gamma, input_range=config.input_range,
        #                       weight_scale=config.weight_scale, bias_data=config.bias_data,
        #                       k=config.k, dist=config.dist, l=config.l, w_e=None if config.unique_w_e else W_e,
        #                       data=config.train_features, data_len=config.train_lens,
        #                       examples_per_cat=config.examples_per_cat, min_len=config.train_min_len)

        if config.data_kernel == 'imagenet':
            mean_max_prob = None
            mean_target_prob = None
            val_mean_max_prob = None
            val_mean_target_prob = None
        else:
            mean_max_prob, mean_target_prob = get_mean_probs(eval_data)
            val_mean_max_prob, val_mean_target_prob = get_mean_probs(val_data)

        # evaluate gd and gd++ performance on evaluation data
        if config.analyze:
            if config.data_kernel == 'imagenet' and config.model_kernel == 'rbf':
                data_idxs = jnp.arange(len(eval_data[0]))
                num_minibatches = len(data_idxs) // config.mb_size

                loss_gd = 0
                acc_gd = 0
                top_3_freq_gd = 0
                prob_dist_gd = None if config.data_kernel == 'imagenet' else 0

                for i in range(num_minibatches):
                    mini_batch = tuple(data[data_idxs[i * config.mb_size:(i + 1) * config.mb_size]] for data in eval_data)

                    loss_gd_temp, _, _, acc_gd_temp, top_3_freq_gd_temp, prob_dist_gd_temp = predict_test.apply(
                        eval_rng,
                        mini_batch,
                        True)
                    loss_gd += loss_gd_temp
                    acc_gd += acc_gd_temp
                    top_3_freq_gd += top_3_freq_gd_temp
                    if config.data_kernel != 'imagenet':
                        prob_dist_gd += prob_dist_gd_temp

                loss_gd /= num_minibatches
                acc_gd /= num_minibatches
                top_3_freq_gd /= num_minibatches
                if config.data_kernel != 'imagenet':
                    prob_dist_gd /= num_minibatches
            else:
                loss_gd, _, _, acc_gd, top_3_freq_gd, prob_dist_gd = predict_test.apply(params_gd, eval_rng, eval_data,
                                                                                True)

        # print("Initial Trained TF params:\n", train_state.params)
        print("Starting Trained TF training...")

        # initialize variables for early stopping
        train_stop = False
        waiting_time = -1
        patience = 15
        best_params_trained = train_state.params
        best_step = 0
        best_val_loss = jnp.inf
        best_val_acc = 0
        best_val_mse = jnp.inf
        best_val_top_3_freq = 0

        # print("GD params:\n")
        # for param in params_gd:
        #     with jnp.printoptions(threshold=sys.maxsize):
        #         print(param)
        #         print(params_gd[param])

        # initialize Trained TF params with GD params
        if config.gd_init:
            tf_init = train_state.params.copy()

            for module in tf_init:
                tf_init[module]['w'] = params_gd[module.replace("transformer", "Transformer_gd")]['w']

                if 'b' in tf_init[module]:
                    tf_init[module]['b'] = params_gd[module.replace("transformer", "Transformer_gd")]['b']

            train_state = TrainState(
                            params=tf_init,
                            opt_state=train_state.opt_state,
                            rng=train_state.rng,
                            step=train_state.step)

        # print("Trained TF params:\n")
        # for param in train_state.params:
        #     if 'key' in param:
        #         print("Trained TF: \n")
        #         print(param)
        #         print(train_state.params[param])
        #
        #         print("\nGD: \n")
        #         print(params_gd[param.replace("transformer", "Transformer_gd")])

        # trained transformer training loop
        original_data_rng = train_data_rng

        tf_training_steps = 500 if config.data_kernel == 'imagenet' else config.training_steps
        for step in range(tf_training_steps):
            if config.cycle_data > 0:
                if step % config.cycle_data == 0:
                    train_data_rng = original_data_rng

            # generate training data
            # rng, train_data_rng = jax.random.split(train_data_rng, 2)

            train_data = get_data(data_creator, rng=train_data_rng, kernel=config.data_kernel,
                                  batch_size=config.bs_train, input_size=config.input_size, dataset_size=config.dataset_size,
                                  embedding_size=config.e_size, cats=config.cats, j=config.j,
                                  sigma=config.sigma, gamma=config.gamma, input_range=config.input_range,
                                  weight_scale=config.weight_scale, bias_data=config.bias_data,
                                  k=config.k, dist=config.dist, l=config.l, w_e=None if config.unique_w_e else W_e,
                                  data=config.train_features, data_len=config.train_lens,
                                  examples_per_cat=config.examples_per_cat, min_len=config.train_min_len)

            rng, _ = jax.random.split(rng, 2)

            data_idxs = jax.random.permutation(rng, len(train_data[0]))
            num_minibatches = len(data_idxs) // config.mb_size

            for i in range(num_minibatches):
                mini_batch = tuple(data[data_idxs[i * config.mb_size:(i + 1) * config.mb_size]] for data in train_data)

                # update transformer
                train_state, metrics = update(train_state, mini_batch, optimizer, constructed=False)

            if len(data_idxs) % config.mb_size != 0:
                mini_batch = tuple(data[data_idxs[(i+1) * config.mb_size:]] for data in train_data)

                # update transformer
                train_state, metrics = update(train_state, mini_batch, optimizer, constructed=False)

            # update transformer
            # train_state, metrics = update(train_state, train_data, optimizer, constructed=False)

            # save and print results every 100 steps
            if step % 100 == 0:
                if config.data_kernel == 'imagenet' and config.model_kernel == 'rbf':
                    data_idxs = jnp.arange(len(val_data[0]))
                    num_minibatches = len(data_idxs) // config.mb_size

                    val_loss = 0
                    val_acc = 0
                    val_top_3_freq = 0
                    val_prob_dist = None if config.data_kernel == 'imagenet' else 0

                    for i in range(num_minibatches):
                        mini_batch = tuple(
                            data[data_idxs[i * config.mb_size:(i + 1) * config.mb_size]] for data in val_data)

                        val_losses_temp, _, _, val_acc_temp, val_top_3_freq_temp, val_prob_dist_temp = predict_test.apply(
                            train_state.params,
                            val_rng, mini_batch,
                            False)
                        val_loss += val_losses_temp
                        val_acc += val_acc_temp
                        val_top_3_freq += val_top_3_freq_temp
                        if config.data_kernel != 'imagenet':
                            val_prob_dist += val_prob_dist_temp

                    val_loss /= num_minibatches
                    val_acc /= num_minibatches
                    val_top_3_freq /= num_minibatches
                    if config.data_kernel != 'imagenet':
                        val_prob_dist /= num_minibatches

                    loss_trans = 0
                    acc_trans = 0
                    top_3_freq_trans = 0
                    prob_dist_trans = None if config.data_kernel == 'imagenet' else 0

                    for i in range(num_minibatches):
                        mini_batch = tuple(
                            data[data_idxs[i * config.mb_size:(i + 1) * config.mb_size]] for data in eval_data)

                        loss_trans_temp, _, _, acc_trans_temp, top_3_freq_trans_temp, prob_dist_trans_temp = predict_test.apply(
                            train_state.params,
                            eval_rng, mini_batch,
                            False)
                        loss_trans += loss_trans_temp
                        acc_trans += acc_trans_temp
                        top_3_freq_trans += top_3_freq_trans_temp
                        if config.data_kernel != 'imagenet':
                            prob_dist_trans += prob_dist_trans_temp

                    loss_trans /= num_minibatches
                    acc_trans /= num_minibatches
                    top_3_freq_trans /= num_minibatches
                    if config.data_kernel != 'imagenet':
                        prob_dist_trans /= num_minibatches
                else:
                    # get validation metrics
                    val_loss, _, _, val_acc, val_top_3_freq, val_prob_dist = predict_test.apply(train_state.params,
                                                                                                val_rng,
                                                                                                val_data,
                                                                                                False)

                    # get testing metrics
                    loss_trans, _, _, acc_trans, top_3_freq_trans, prob_dist_trans = predict_test.apply(train_state.params,
                                                                                                        eval_rng,
                                                                                                        eval_data, False)

                loss_trans_list[cur_seed].append(loss_trans)
                acc_trans_list[cur_seed].append(acc_trans)
                top_3_freq_trans_list[cur_seed].append(top_3_freq_trans)
                prob_dist_trans_list[cur_seed].append(prob_dist_trans)
                loss_trans_train_list[cur_seed].append(metrics['train_loss'].item(), )
                max_prob_list[cur_seed].append(mean_max_prob)
                target_prob_list[cur_seed].append(mean_target_prob)

                if config.analyze:
                    # append gd metrics to list
                    losses_gd_list[cur_seed].append(loss_gd)
                    acc_gd_list[cur_seed].append(acc_gd)
                    top_3_freq_gd_list[cur_seed].append(top_3_freq_gd)
                    prob_dist_gd_list[cur_seed].append(prob_dist_gd)

                    if config.data_kernel == 'imagenet':
                        display(("Current seed", cur_seed,
                                 "Training step", step, "Gradient descent loss", loss_gd.item(),
                                 "Trained TF loss", loss_trans.item(),
                                 "GD Accuracy", acc_gd.item(),
                                 "Trained TF Accuracy", acc_trans.item(),
                                 "GD Top {} Frequency".format(config.top_preds), top_3_freq_gd.item(),
                                 "Trained TF Top {} Frequency".format(config.top_preds), top_3_freq_trans.item()),
                                display_id="Cur met")
                    else:
                        display(("Current seed", cur_seed,
                                 "Training step", step, "Gradient descent loss", loss_gd.item(),
                                 "Trained TF loss", loss_trans.item(),
                                 "GD Accuracy", acc_gd.item(),
                                 "Trained TF Accuracy", acc_trans.item(),
                                 "GD Top {} Frequency".format(config.top_preds), top_3_freq_gd.item(),
                                 "Trained TF Top {} Frequency".format(config.top_preds), top_3_freq_trans.item(),
                                 "GD Probability Loss", prob_dist_gd.item(),
                                 "Trained TF Probability Loss", prob_dist_trans.item(),
                                 "Mean Max Probability", mean_max_prob.item(),
                                 "Mean Target Probability", mean_target_prob.item()),
                                display_id="Cur met")
                else:
                    print(step, loss_trans)

                # early stopping
                if config.early_stopping and not train_stop:
                    if val_loss < best_val_loss:
                        best_params_trained = train_state.params.copy()
                        best_step = step
                        best_val_loss = val_loss
                        best_val_acc = val_acc
                        best_val_mse = val_prob_dist
                        best_val_top_3_freq = val_top_3_freq

                        waiting_time = 0
                    else:
                        waiting_time += 1

            # if config.early_stopping and waiting_time > patience:
            #     train_stop = True
            #     break

        # record best training step on validation set
        best_idx_trans_list.append(best_step)

        # save trained transformer parameters
        with open('{}/{}/params_trained_seed_{}.pickle'.format(config.save_folder, config.model_kernel, cur_seed),
                  'wb') as handle:
            pickle.dump(train_state.params, handle, protocol=5)

        if early_stopping:
            with open('{}/{}/params_trained_es_seed_{}.pickle'.format(config.save_folder, config.model_kernel, cur_seed),
                      'wb') as handle:
                pickle.dump(best_params_trained, handle, protocol=5)

        # for module in train_state.params:
        #     print(module)
        print("Trained TF training complete")

        # save results into JSON
        try:
            results = get_results_dict(loss_trans_list, loss_trans_train_list, acc_trans_list, top_3_freq_trans_list,
                                       prob_dist_trans_list, best_idx_trans_list,
                                       losses_gd_list, acc_gd_list, top_3_freq_gd_list, prob_dist_gd_list,
                                       max_prob_list, target_prob_list,
                                       cos_sim_list, grad_norm_list, p_norm_list,
                                       gd_train_eval_loss_list, gd_train_eval_acc_list, gd_train_eval_top_3_freq_list,
                                       gd_train_eval_prob_dist_list, gd_train_eval_max_prob_list,
                                       gd_train_eval_target_prob_list, gd_val_loss_list, gd_val_acc_list,
                                       gd_val_top_3_freq_list, gd_val_prob_dist_list, gd_val_max_prob_list,
                                       gd_val_target_prob_list, gd_val_best_steps)

            with open(config.save_folder + "/" + config.model_kernel + "/results.json", "w") as fp:
                json.dump(results, fp)

            print("Done writing results into json file")
        except Exception as e:
            print("Error getting results")
            print(e)

    config.train_features = None
    config.val_features = None
    config.train_lens = None
    config.val_lens = None

    # save config into JSON
    with open(config.save_folder + "/" + config.model_kernel + "/config.json", "w") as fp:
        json.dump(config.to_dict(), fp)

    print("Done saving config into json file")

    # display metrics

    # Loss of GD + Trained TF on Test Data
    plot.display_training(config.save_folder, losses_gd_list, trained_tf=loss_trans_list,
                          num_iter_os=len(loss_trans_list[0]) * 100,
                          plot_title="Trained TF Loss",
                          title="{}/trained_tf_loss".format(config.model_kernel), single_seeds_gd=False,
                          single_seeds_tf=True,
                          x_label='Training Steps',
                          y_label='Negative Log-Likelihood', yscale_log=False, y_lim_l=0,
                          y_lim_u=1.5,
                          color_add=0, loc_first='best', width=6, height=4.5)

    # Accuracy of GD + Trained TF on Test Data
    plot.display_training(config.save_folder, acc_gd_list, trained_tf=acc_trans_list,
                          num_iter_os=len(acc_trans_list[0]) * 100,
                          plot_title="Trained TF Accuracy",
                          title="{}/trained_tf_acc".format(config.model_kernel), single_seeds_gd=False,
                          single_seeds_tf=True,
                          x_label='Training Steps',
                          y_label='Accuracy', yscale_log=False, y_lim_l=0,
                          y_lim_u=1,
                          color_add=0, loc_first='best', width=6, height=4.5)

    # Top 3 Frequency of GD + Trained TF on Test Data
    plot.display_training(config.save_folder, top_3_freq_gd_list, trained_tf=top_3_freq_trans_list,
                          num_iter_os=len(top_3_freq_trans_list[0]) * 100,
                          plot_title="Trained TF Top 3 Frequency",
                          title="{}/trained_tf_top_3_freq".format(config.model_kernel), single_seeds_gd=False,
                          single_seeds_tf=True,
                          x_label='Training Steps',
                          y_label='Top 3 Frequency', yscale_log=False, y_lim_l=0,
                          y_lim_u=1,
                          color_add=0, loc_first='best', width=6, height=4.5)

    # MSE on Category Probabilities of GD + Trained TF on Test Data
    if config.data_kernel != "imagenet":
        plot.display_training(config.save_folder, prob_dist_gd_list, trained_tf=prob_dist_trans_list,
                              num_iter_os=len(top_3_freq_trans_list[0]) * 100,
                              plot_title="Trained TF MSE on Category Probabilities",
                              title="{}/trained_tf_prob_mse".format(config.model_kernel), single_seeds_gd=False,
                              single_seeds_tf=True,
                              x_label='Training Steps',
                              y_label='MSE on Category Probabilities', yscale_log=False, y_lim_l=0,
                              y_lim_u=0.1,
                              color_add=0, loc_first='best', width=6, height=4.5)


if __name__ == '__main__':
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = '\"false\"'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '\"0.99\"'

    print(jax.default_backend())

    # Checking the number of devices we have here
    num_devices = len(jax.local_devices())
    print(f"Running on {num_devices} devices: \n\n{jax.local_devices()}")

    # Create a Sharding object to distribute a value across devices:
    sharding = PositionalSharding(mesh_utils.create_device_mesh((2,)))

    # Create an array of random values:
    x = jax.random.normal(jax.random.key(0), (8192, 8192))

    # and use jax.device_put to distribute it across devices:
    y = jax.device_put(x, sharding.reshape(2, 1))
    jax.debug.visualize_array_sharding(y)

    args = parser.parse_args()

    run_experiment(args.save_path,
                   args.num_seeds,
                   args.num_layers,
                   args.recurrent,
                   args.data_kernel,
                   args.model_kernel,
                   args.categories,
                   args.bias_head,
                   args.bias_data,
                   args.gamma,
                   args.sigma,
                   args.dataset_size,
                   args.input_size,
                   args.embedding_size,
                   args.j,
                   args.k,
                   args.l,
                   args.dist,
                   args.one_hot_emb,
                   args.num_heads_gd,
                   args.num_heads_tr,
                   args.gd_plus,
                   args.diag,
                   args.early_stopping,
                   args.unique_w_e,
                   args.examples_per_cat,
                   args.training_data_size,
                   args.minibatch_size,
                   args.training_steps_tf,
                   args.training_steps_gd,
                   args.holdout,
                   args.gd_init,
                   args.linear_approx,
                   args.use_mlp)
