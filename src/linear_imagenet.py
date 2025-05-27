"""
Trains an MLP on ImageNet data.
"""

import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

import matplotlib.pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm

import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

import pickle
import json

from transformer import Transformer
from data import create_img_data_linear, create_weights, create_w_e
from config import config
import train_linear
from util import *
import plot

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
               "top_{}_freq_trans_list".format(config.top_preds): [[y.tolist() for y in x if y is not None] for x in top_3_freq_trans_list],
               'prob_dist_trans_list': [[y.tolist() for y in x if y is not None] for x in prob_dist_trans_list],
               'best_idx_trans_list': best_idx_trans_list,
               'losses_gd_list': [[y.tolist() for y in x if y is not None] for x in losses_gd_list],
               'acc_gd_list': [[y.tolist() for y in x if y is not None] for x in acc_gd_list],
               'top_{}_freq_gd_list'.format(config.top_preds): [[y.tolist() for y in x if y is not None] for x in top_3_freq_gd_list],
               'prob_dist_gd_list': [[y.tolist() for y in x if y is not None] for x in prob_dist_gd_list],
               'max_prob_list': [[y.tolist() for y in x if y is not None] for x in max_prob_list],
               'target_prob_list': [[y.tolist() for y in x if y is not None] for x in target_prob_list],
               'cos_sim_list': [[y.tolist() for y in x if y is not None] for x in cos_sim_list],
               'grad_norm_list': [[y.tolist() for y in x if y is not None] for x in grad_norm_list],
               'p_norm_list': [[y.tolist() for y in x if y is not None] for x in p_norm_list],
               'gd_train_loss_list': [[y.tolist() for y in x if y is not None] for x in gd_train_loss_list],
               'gd_train_acc_list': [[y.tolist() for y in x if y is not None] for x in gd_train_acc_list],
               "gd_train_top_{}_freq_list".format(config.top_preds): [[y.tolist() for y in x if y is not None] for x in gd_train_top_3_freq_list],
               'gd_train_prob_dist_list': [[y.tolist() for y in x if y is not None] for x in gd_train_prob_dist_list],
               'gd_train_max_prob_list': [[y.tolist() for y in x if y is not None] for x in gd_train_max_prob_list],
               'gd_train_target_prob_list': [[y.tolist() for y in x if y is not None] for x in
                                             gd_train_target_prob_list],
               'gd_val_loss_list': [[y.tolist() for y in x if y is not None] for x in gd_val_loss_list],
               'gd_val_acc_list': [[y.tolist() for y in x if y is not None] for x in gd_val_acc_list],
               "gd_val_top_{}_freq_list".format(config.top_preds): [[y.tolist() for y in x if y is not None] for x in gd_val_top_3_freq_list],
               'gd_val_prob_dist_list': [[y.tolist() for y in x if y is not None] for x in gd_val_prob_dist_list],
               'gd_val_max_prob_list': [[y.tolist() for y in x if y is not None] for x in gd_val_max_prob_list],
               'gd_val_target_prob_list': [[y.tolist() for y in x if y is not None] for x in gd_val_target_prob_list],
               'gd_val_best_step': [x for x in gd_val_best_steps]
               }

    return results


def get_data(data_creator, rng, kernel, batch_size, input_size, dataset_size, embedding_size, cats, j, sigma, gamma,
             input_range, weight_scale, bias_data, k, dist, l, w_e, data, data_len, examples_per_cat, min_len):
    """Loads data."""
    if kernel == 'imagenet':
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
                   minibatch_size=512, holdout=False, gd_init=False,
                   linear_approx=False):
    """Run experiments."""
    pl.rcParams.update({'font.size': 12})
    pl.rc('axes', labelsize=14)
    pl.rcParams.update({
        "text.usetex": False,
    })

    # initialize experiment parameters
    conf_init(save_path, num_seeds, num_layers, recurrent, data_kernel, modeL_kernel, categories, use_bias_head,
              use_bias_data, gamma, sigma, dataset_size, input_size, embedding_size, j, k, l, dist, one_hot_emb, num_heads_gd,
              num_heads_tr, gd_plus, diag, early_stopping, unique_w_e, examples_per_cat,
              training_data_size=training_data_size, minibatch_size=minibatch_size, hold_out=holdout,
              gd_init=gd_init, linear_approx=linear_approx, num_queries=1)

    # set dataloader
    if config.data_kernel == 'imagenet':
        data_creator = vmap(create_img_data_linear, in_axes=(0, None, None, None, None, None, None), out_axes=0)

    train_linear.change_dataloader()

    # initialize lists to store metrics
    loss_trans_list = [[] for _ in range(config.bs_eval)]
    acc_trans_list = [[] for _ in range(config.bs_eval)]
    top_3_freq_trans_list = [[] for _ in range(config.bs_eval)]
    prob_dist_trans_list = [[] for _ in range(config.bs_eval)]

    loss_trans_train_list = [[] for _ in range(config.bs_eval)]

    losses_gd_list = [[] for _ in range(config.bs_eval)]
    acc_gd_list = [[] for _ in range(config.bs_eval)]
    top_3_freq_gd_list = [[] for _ in range(config.bs_eval)]
    prob_dist_gd_list = [[] for _ in range(config.bs_eval)]

    max_prob_list = [[] for _ in range(config.bs_eval)]
    target_prob_list = [[] for _ in range(config.bs_eval)]

    val_losses = [[] for _ in range(config.bs_eval)]
    val_acc = [[] for _ in range(config.bs_eval)]
    val_top_3_freq = [[] for _ in range(config.bs_eval)]
    val_prob_dist = [[] for _ in range(config.bs_eval)]
    val_mean_max_prob_list = [[] for _ in range(config.bs_eval)]
    val_mean_target_prob_list = [[] for _ in range(config.bs_eval)]
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

    cos_sim_list = [[] for _ in range(config.bs_eval)]
    grad_norm_list = [[] for _ in range(config.bs_eval)]
    p_norm_list = [[] for _ in range(config.bs_eval)]

    print("gamma: ", config.gamma)
    print("sigma: ", config.sigma)
    print("dataset size: ", config.dataset_size)
    print("training batch size: ", config.bs_eval)
    print("lambda: ", config.l)
    print("k: ", config.k)
    print("input size: ", config.input_size)
    print("data embedding size: ", config.data_e_size)
    print("params embedding size: ", config.e_size)
    print("gd lr: ", config.gd_lr)
    print("trained tf lr: ", config.lr)
    print("data kernel: ", config.data_kernel)
    print("model kernel: ", config.model_kernel)

    # create save directory
    if not os.path.isdir(config.save_folder):
        os.mkdir(config.save_folder)

    if not os.path.isdir(config.save_folder + "/" + config.model_kernel):
        os.mkdir(config.save_folder + "/" + config.model_kernel)

    # begin training
    eval_rng = jax.random.PRNGKey(5)
    eval_rng, val_rng, train_data_rng, w_e_rng = jax.random.split(eval_rng, num=4)

    W_e = create_w_e(w_e_rng, config.data_e_size, config.cats)

    print("W_e mean: ", jnp.mean(W_e, axis=1))
    w_e_percentage = 100 * jnp.abs(jnp.expand_dims(jnp.mean(W_e, axis=1), axis=-1) / W_e)
    print("Percentage of original W_e: ", w_e_percentage)

    w_e_data = {'W_e': [x.tolist() for x in W_e],
                'W_e_mean': jnp.mean(W_e, axis=1).tolist(),
                'W_e_percentage': [x.tolist() for x in w_e_percentage]}

    with open(config.save_folder + "/" + config.model_kernel + "/w_e_data.json", "w") as fp:
        json.dump(w_e_data, fp)

    for cur_seed in range(config.num_seeds):
        print("Current seed: ", cur_seed)
        config.seed = cur_seed

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
        train_data = get_data(data_creator, rng=train_data_rng, kernel=config.data_kernel,
                              batch_size=config.bs_train, input_size=config.input_size,
                              dataset_size=config.dataset_size,
                              embedding_size=config.e_size, cats=config.cats, j=config.j,
                              sigma=config.sigma, gamma=config.gamma, input_range=config.input_range,
                              weight_scale=config.weight_scale, bias_data=config.bias_data,
                              k=config.k, dist=config.dist, l=config.l, w_e=None if config.unique_w_e else W_e,
                              data=config.train_features, data_len=config.train_lens,
                              examples_per_cat=config.examples_per_cat, min_len=config.train_min_len)

        for cur_dataset in range(config.bs_train):
            optimizer, train_state, _, rng = train_linear.init()
            rng, data_rng = jax.random.split(rng, num=2)

            print("Training Linear NN on Dataset {}".format(cur_dataset))

            # initialize variables for early stopping
            # train_stop = False
            # waiting_time = -1
            # patience = 15
            # best_params_trained = train_state.params
            # best_step = 0
            # best_val_loss = jnp.inf
            # best_val_acc = 0
            # best_val_top_3_freq = 0

            # print("Linear NN params:\n")
            # for param in train_state.params:
            #     print(param)
            #     print(train_state.params[param])

            train_data_linear = tuple(data[cur_dataset, :-1] for data in eval_data)
            eval_data_linear = tuple(data[cur_dataset, -1:] for data in eval_data)

            # trained transformer training loop
            for step in range(config.training_steps_linear):
                # update linear NN
                train_state, metrics = train_linear.update(train_state, train_data_linear, optimizer)

                # save and print results every 100 steps
                if step % 100 == 0:
                    # get testing metrics
                    loss_trans, acc_trans, top_3_freq_trans = train_linear.predict_test.apply(train_state.params,
                                                                                              eval_rng, eval_data_linear)

                    loss_trans_list[cur_dataset].append(loss_trans)
                    acc_trans_list[cur_dataset].append(acc_trans)
                    top_3_freq_trans_list[cur_dataset].append(top_3_freq_trans)
                    loss_trans_train_list[cur_dataset].append(metrics['train_loss'].item(), )

                    if config.analyze:
                        display(("Current seed", cur_seed,
                                 "Current dataset", cur_dataset,
                                 "Training step", step,
                                 "Linear NN loss", loss_trans.item(),
                                 "Linear NN Accuracy", acc_trans.item(),
                                 "Linear NN Top {} Frequency".format(config.top_preds), top_3_freq_trans.item()),
                                display_id="Cur met")
                    else:
                        print(step, loss_trans)

                    # early stopping
                    # if config.early_stopping and not train_stop:
                    #     if val_loss < best_val_loss:
                    #         best_params_trained = train_state.params.copy()
                    #         best_step = step
                    #         best_val_loss = val_loss
                    #         best_val_acc = val_acc
                    #         best_val_top_3_freq = val_top_3_freq
                    #
                    #         waiting_time = 0
                    #     else:
                    #         waiting_time += 1

                # if config.early_stopping and waiting_time > patience:
                #     train_stop = True
                #     break

            # record best training step on validation set
            # best_idx_trans_list.append(best_step)

            print("Linear NN training complete")

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


if __name__ == '__main__':
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = '\"false\"'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '\"0.99\"'

    print(jax.default_backend())

    # Checking the number of devices we have here
    num_devices = len(jax.local_devices())
    print(f"Running on {num_devices} devices: \n\n{jax.local_devices()}")

    # Create a Sharding object to distribute a value across devices:
    sharding = PositionalSharding(mesh_utils.create_device_mesh((2,)))

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
                   args.holdout,
                   args.gd_init,
                   args.linear_approx)



