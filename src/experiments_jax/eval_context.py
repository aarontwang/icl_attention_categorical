import sys
import os

import matplotlib.pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import numpy as np

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
parser.add_argument("-s", "--scale_projection", action="store_true", help="scale projection matrix")


def get_results_dict(loss_trans_list, loss_gd_list, acc_trans_list, acc_gd_list, top_3_freq_trans_list,
                     top_3_freq_gd_list, prob_dist_trans_list, prob_dist_gd_list, factors):
    results = {'loss_trans_list': [[y.tolist() for y in x if y is not None] for x in loss_trans_list],
               'loss_gd_list': [[y.tolist() for y in x if y is not None] for x in loss_gd_list],
               'acc_trans_list': [[y.tolist() for y in x if y is not None] for x in acc_trans_list],
               'acc_gd_list': [[y.tolist() for y in x if y is not None] for x in acc_gd_list],
               'top_2_freq_trans_list': [[y.tolist() for y in x if y is not None] for x in top_3_freq_trans_list],
               'top_2_freq_gd_list': [[y.tolist() for y in x if y is not None] for x in top_3_freq_gd_list],
               'prob_dist_trans_list': [[y.tolist() for y in x if y is not None] for x in prob_dist_trans_list],
               'prob_dist_gd_list': [[y.tolist() for y in x if y is not None] for x in prob_dist_gd_list],
               'factors': factors}

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


def change_context(save_path, num_seeds=1, num_layers=1, recurrent=True, data_kernel='linear', modeL_kernel='linear',
                   categories=2, use_bias_head=False, use_bias_data=False,
                   gamma=None, sigma=None, dataset_size=20, input_size=2, embedding_size=5, j=5, k=5, l=3, dist=0.1,
                   one_hot_emb=False, num_heads_gd=1, num_heads_tr=1, gd_plus=False, diag=False,
                   early_stopping=False, unique_w_e=False, examples_per_cat=10, holdout=False, gd_init=False,
                   linear_approx=False, scale_projection=False):
    """Run experiments."""
    pl.rcParams.update({'font.size': 12})
    pl.rc('axes', labelsize=14)
    pl.rcParams.update({
        "text.usetex": False,
    })

    # initialize experiment parameters
    conf_init(save_path, num_seeds, num_layers, recurrent, data_kernel, modeL_kernel, categories, use_bias_head,
              use_bias_data, gamma, sigma, dataset_size, input_size, embedding_size, j, k, l, dist, one_hot_emb,
              num_heads_gd,
              num_heads_tr, gd_plus, diag, early_stopping, unique_w_e, examples_per_cat, holdout,
              gd_init, linear_approx, 1)

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
        # TODO: replace with imagenet data creator code
        data_creator = vmap(create_img_data,
                            in_axes=(0, None, None, None, None, None, None), out_axes=0)

    change_dataloader()

    print("gamma: ", config.gamma)
    print("sigma: ", config.sigma)
    print("dataset size: ", config.dataset_size)
    print("training batch size: ", config.bs_train)
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

    loss_gd_list = []
    loss_trans_list = []
    acc_gd_list = []
    acc_trans_list = []
    top_3_freq_gd_list = []
    top_3_freq_trans_list = []
    prob_dist_gd_list = []
    prob_dist_trans_list = []

    if data_kernel == 'imagenet':
        factors = [2, 3, 10]
    else:
        factors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for cur_seed in range(config.num_seeds):
        print("Current seed: ", cur_seed)
        config.seed = cur_seed

        # load gd transformer parameters
        with open("{}/{}/params_gd_seed_{}.pickle".format(save_path, modeL_kernel, cur_seed), "rb") as f:
            params_gd = pickle.load(f)

        # load trained tf parameters
        with open("{}/{}/params_trained_seed_{}.pickle".format(save_path, modeL_kernel, cur_seed), "rb") as f:
            params_tf = pickle.load(f)

        print("Loaded parameters")

        loss_trans_list.append([])
        loss_gd_list.append([])
        prob_dist_gd_list.append([])
        prob_dist_trans_list.append([])
        acc_gd_list.append([])
        acc_trans_list.append([])
        top_3_freq_gd_list.append([])
        top_3_freq_trans_list.append([])

        for factor in factors:
            if scale_projection and config.model_kernel != 'softmax':
                print("scaling")
                projection_matrix = jnp.identity(config.input_size + 2 * config.e_size) * \
                                    config.dataset_size / (factor * config.cats)
                for module in params_tf:
                    if 'linear' in module:
                        params_tf[module]['w'] = projection_matrix
                        # print("New Trained TF projection ({}): ".format(factor), params_tf[module]['w'])

                for module in params_gd:
                    if 'linear' in module:
                        params_gd[module]['w'] = projection_matrix
                        # print("New GD projection ({}): ".format(factor), params_gd[module]['w'])

            eval_data = get_data(data_creator, rng=eval_rng, kernel=config.data_kernel,
                                 batch_size=config.bs_eval, input_size=config.input_size,
                                 dataset_size=factor * config.cats,
                                 embedding_size=config.e_size, cats=config.cats, j=config.j,
                                 sigma=config.sigma, gamma=config.gamma, input_range=config.input_range,
                                 weight_scale=config.weight_scale, bias_data=config.bias_data,
                                 k=config.k, dist=config.dist, l=config.l, w_e=None if config.unique_w_e else W_e,
                                 data=config.val_features, data_len=config.val_lens,
                                 examples_per_cat=factor, min_len=config.val_min_len)

            if data_kernel != 'imagenet':
                mean_max_prob, mean_target_prob = get_mean_probs(eval_data)

            if modeL_kernel == 'rbf':
                loss_gd_temp = []
                acc_gd_temp = []
                top_3_freq_gd_temp = []
                prob_dist_gd_temp = []

                loss_trans_temp = []
                acc_trans_temp = []
                top_3_freq_trans_temp = []
                prob_dist_trans_temp = []

                batch_size = config.bs_eval // 128
                for i in range(128):
                    mini_batch = tuple(data[i*batch_size:(i+1)*batch_size] for data in eval_data)
                    loss_gd, _, _, acc_gd, top_3_freq_gd, prob_dist_gd = predict_test.apply(params_gd, eval_rng, mini_batch,
                                                                                            True)
                    loss_trans, _, _, acc_trans, top_3_freq_trans, prob_dist_trans = predict_test.apply(params_tf, eval_rng,
                                                                                                        mini_batch, False)

                    loss_gd_temp.append(loss_gd)
                    acc_gd_temp.append(acc_gd)
                    prob_dist_gd_temp.append(prob_dist_gd)
                    top_3_freq_gd_temp.append(top_3_freq_gd)

                    loss_trans_temp.append(loss_trans)
                    acc_trans_temp.append(acc_trans)
                    prob_dist_trans_temp.append(prob_dist_trans)
                    top_3_freq_trans_temp.append(top_3_freq_trans)

                loss_gd = np.mean(loss_gd_temp)
                acc_gd = np.mean(acc_gd_temp)
                top_3_freq_gd = np.mean(top_3_freq_gd_temp)
                prob_dist_gd = np.mean(prob_dist_gd_temp)

                loss_trans = np.mean(loss_trans_temp)
                acc_trans = np.mean(acc_gd_temp)
                top_3_freq_trans = np.mean(top_3_freq_trans_temp)
                prob_dist_trans = np.mean(prob_dist_gd_temp)
            else:

                loss_gd, _, _, acc_gd, top_3_freq_gd, prob_dist_gd = predict_test.apply(params_gd, eval_rng, eval_data,
                                                                                        True)
                loss_trans, _, _, acc_trans, top_3_freq_trans, prob_dist_trans = predict_test.apply(params_tf, eval_rng,
                                                                                                    eval_data, False)

            if data_kernel == 'imagenet':
                display(("Factor", factor,
                         "GD loss", loss_gd.item(),
                         "Trained TF loss", loss_trans.item(),
                         "GD Accuracy", acc_gd.item(),
                         "Trained TF Accuracy", acc_trans.item(),
                         "GD Top {} Frequency".format(config.top_preds), top_3_freq_gd.item(),
                         "Trained TF Top {} Frequency".format(config.top_preds), top_3_freq_trans.item()),
                        display_id="Cur met")
            else:
                display(("Factor", factor,
                         "GD loss", loss_gd.item(),
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

            loss_gd_list[-1].append(loss_gd)
            loss_trans_list[-1].append(loss_trans)
            acc_gd_list[-1].append(acc_gd)
            acc_trans_list[-1].append(acc_trans)
            top_3_freq_gd_list[-1].append(top_3_freq_gd)
            top_3_freq_trans_list[-1].append(top_3_freq_trans)
            prob_dist_gd_list[-1].append(prob_dist_gd)
            prob_dist_trans_list[-1].append(prob_dist_trans)

    config.train_features = None
    config.val_features = None
    config.train_lens = None
    config.val_lens = None

    # save into JSON
    try:
        results = get_results_dict(loss_trans_list, loss_gd_list, acc_trans_list, acc_gd_list, top_3_freq_trans_list,
                                   top_3_freq_gd_list, prob_dist_trans_list, prob_dist_gd_list, factors)

        if scale_projection:
            with open(config.save_folder + "/" + config.model_kernel + "/context_results_scale.json", "w") as fp:
                json.dump(results, fp)
        else:
            with open(config.save_folder + "/" + config.model_kernel + "/context_results_no_scale.json", "w") as fp:
                json.dump(results, fp)

        print("Done writing results into json file")
    except Exception as e:
        print("Error writing results")
        print(e)


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

    change_context(args.save_path,
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
                   args.holdout,
                   args.gd_init,
                   args.linear_approx,
                   args.scale_projection)
