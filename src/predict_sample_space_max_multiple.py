import sys
import os

import matplotlib.pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm

import pickle

from transformer import Transformer
from data import create_cat_data_rbf, create_cat_data_grid, create_cat_data_random_grid, create_weights
from config import config
from train import *
from eval import *
from util import *
import plot

from IPython.display import display

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="directory with model parameters")
parser.add_argument("save_path", help="directory to save results to")
parser.add_argument("num_seeds", type=int, help="number of seeds")


def load_weights(save_path, kernel, seed, gd=False):
    with open("{}/{}/params_{}_seed_{}.pickle".format(save_path, kernel, "gd" if gd else "trained_es", seed), "rb") as f:
        weights = pickle.load(f)
        print(weights)

    return weights


def gen_figs(data_path, save_path, num_seeds):
    kernels = ['linear', 'exp', 'rbf', 'laplacian', 'softmax']

    seeds = [0, 1, 2, 3, 4]

    for kernel in kernels:
        conf_init(data_path, num_seeds, 1, True, "random_grid", kernel, 20, False,
                  False, 1, 1, 100, 5, 5, 5, 3, 0.1, False, 1,
                  1, False, False, False, False, True, False, 101**2)

        change_dataloader()

        for cur_seed in seeds:
            params_gd = load_weights(data_path, kernel, 0, gd=True)
            params_tf = load_weights(data_path, kernel, 0, gd=False)

            # begin training
            eval_rng = jax.random.PRNGKey(5)

            print("Current seed: ", cur_seed)
            config.seed = cur_seed
            rng = jax.random.PRNGKey(config.seed)
            rng, train_rng = jax.random.split(rng, 2)
            rng, data_rng, val_rng = jax.random.split(rng, 3)
            rng, scan_rng = jax.random.split(rng, 2)

            if not os.path.exists("{}/{}".format(save_path, kernel)):
                os.mkdir("{}/{}".format(save_path, kernel))

            if not os.path.exists("{}/{}/gd".format(save_path, kernel)):
                os.mkdir("{}/{}/gd".format(save_path, kernel))

            if not os.path.exists("{}/{}/trained".format(save_path, kernel)):
                os.mkdir("{}/{}/trained".format(save_path, kernel))

            if not os.path.exists("{}/{}/gd/seed_{}".format(save_path, kernel, cur_seed)):
                os.mkdir("{}/{}/gd/seed_{}".format(save_path, kernel, cur_seed))

            if not os.path.exists("{}/{}/trained/seed_{}".format(save_path, kernel, cur_seed)):
                os.mkdir("{}/{}/trained/seed_{}".format(save_path, kernel, cur_seed))

            predict_sample_space(scan_rng, "{}/{}".format(save_path, kernel), cur_seed, params_tf, params_gd)


def predict_sample_space(rng, save_path, seed, params_tf, params_gd):
    # Define range and step size
    x_test = np.arange(-1, 1.02, 0.02)
    y_test = np.arange(-1, 1.02, 0.02)

    # Generate grid
    xx, yy = np.meshgrid(x_test, y_test)

    x_query = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    rng, new_rng = jax.random.split(rng, 2)

    scan_data = gen_scan_data(rng, x_query)

    # get in-context data
    c_idxs = scan_data[0]
    max_c_idxs = scan_data[1]
    x = scan_data[2]
    x_prob = scan_data[6].squeeze()[0]
    x_labels = scan_data[3].squeeze()

    full_preds_gd = jnp.squeeze(predict.apply(params_gd, new_rng, scan_data[4], gd=True))
    full_preds_tf = jnp.squeeze(predict.apply(params_tf, new_rng, scan_data[4], gd=False))

    if not os.path.exists("{}".format(save_path)):
        os.mkdir("{}".format(save_path))

    if not os.path.exists("{}/trained".format(save_path)):
        os.mkdir("{}/trained".format(save_path))

    if not os.path.exists("{}/trained".format(save_path)):
        os.mkdir("{}/trained".format(save_path))

    if not os.path.exists("{}/gd/seed_{}".format(save_path, seed)):
        os.mkdir("{}/gd/seed_{}".format(save_path, seed))

    if not os.path.exists("{}/trained/seed_{}".format(save_path, seed)):
        os.mkdir("{}/trained/seed_{}".format(save_path, seed))

    plot_scan("{}".format(save_path), seed, x_query, full_preds_tf, full_preds_gd, x, x_labels, max_c_idxs)


def gen_scan_data(rng, x_query):
    c_size = 100
    i_size = 2
    e_size = 5
    cats = 20
    input_range = 1

    rng, new_rng, new_rng2, new_rng3, new_rng4, new_rng5, new_rng6 = jax.random.split(rng, num=7)
    W_e = jax.random.normal(new_rng, shape=(e_size, cats))

    # draw training data and query
    x = jax.random.uniform(new_rng2, shape=[c_size, i_size],
                           minval=-input_range, maxval=input_range)

    c_idx = jax.random.choice(new_rng4, a=cats, shape=(4,), replace=False)
    c = W_e[:, c_idx]

    quad_logits = c.T @ W_e
    max_quad_idx = jnp.argmax(quad_logits, axis=1)

    x_0 = x[:, 0]
    x_1 = x[:, 1]

    temp_0 = jnp.where(x_0 >= 0, 2, 0)
    temp_1 = jnp.where(x_1 >= 0, 3, 2)
    temp = jnp.concatenate([temp_0[:, None], temp_1[:, None]], axis=1)
    quad = jnp.sum(temp, axis=1) - 2

    # calculate f
    f = c[:, quad]

    # calculate the probability logits of each class
    probs = jax.nn.softmax(f.T @ W_e, axis=1)

    # randomly draw labels for each sample
    y_data = jax.random.categorical(new_rng5, f.T @ W_e, axis=1)

    v_data_full = jax.nn.one_hot(y_data, num_classes=cats)

    # get quadrant of x_query
    x_0_query = x_query[:, 0]
    x_1_query = x_query[:, 1]
    temp_0_query = jnp.where(x_0_query >= 0, 2, 0)
    temp_1_query = jnp.where(x_1_query >= 0, 3, 2)
    temp_query = jnp.concatenate([temp_0_query[:, None], temp_1_query[:, None]], axis=1)
    quad_query = jnp.sum(temp_query, axis=1) - 2

    # calculate f(x_query)
    f_target = c[:, quad_query]

    # calculate the probability logits of each class for the query
    probs_target = jax.nn.softmax(f_target.T @ W_e, axis=1)

    # randomly draw label for query
    y_target = jax.random.categorical(new_rng6, f_target.T @ W_e, axis=1)

    v_target_full = jax.nn.one_hot(y_target, num_classes=cats)

    W_e_seq = W_e[:, y_data].T
    E_w_e_init = jnp.zeros(shape=(c_size, e_size))
    f_init = jnp.zeros(shape=(c_size, e_size))

    seq = jnp.concatenate([x, v_data_full - 1 / cats, f_init], axis=-1)

    target = jnp.concatenate([x_query, v_target_full - 1 / cats, f_target.T], axis=-1)

    zero = jnp.concatenate([x_query, jnp.zeros((x_query.shape[0], cats)) - 1 / cats,
                                             jnp.zeros((x_query.shape[0], e_size))], axis=-1)

    seq = jnp.concatenate([seq, zero], axis=0)

    return jnp.squeeze(c_idx), jnp.squeeze(max_quad_idx), jnp.squeeze(x), jnp.squeeze(y_data), \
        jnp.expand_dims(seq, axis=0), \
        jnp.squeeze(target), probs, \
        jnp.squeeze(probs_target), jnp.squeeze(v_data_full), jnp.squeeze(v_target_full)


def plot_scan(save_path, seed, x_query, preds_tf, preds_gd, x, x_label, c_idxs):
    cm1 = mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"])

    # In-context examples
    c_size = x.shape[0]

    # Quadrant 1 - Trained TF
    max_q1 = max(max(preds_tf[:, int(c_idxs[3])]), max(preds_gd[:, int(c_idxs[3])]))

    fig1 = plt.figure()
    fig1.set_size_inches(8, 4.8)
    ax1 = plt.subplot(111)

    colors_1_tf = cm1(preds_tf[:, int(c_idxs[3])])
    ax1.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_tf[:, int(c_idxs[3])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    legend_used = []

    ax1.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])), marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax1.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax1.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax1.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    legend_used.append(int(c_idxs[3]))
    ax1.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q1)".format(int(c_idxs[3])))

    # Shrink current axis by 20%
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])

    fig1.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q1), cmap=cm1),
                  ax=ax1, orientation='vertical',
                  label='Probability of Category {} (Q1)'.format(int(c_idxs[3])))

    ax1.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax1.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/trained/seed_{}/seed_{}_quad_1.png'.format(save_path, seed, seed))
    plt.close(fig1)

    # Quadrant 1 - GD

    fig2 = plt.figure()
    fig2.set_size_inches(8, 4.8)
    ax2 = plt.subplot(111)

    legend_used = []

    colors_1_gd = cm1(preds_gd[:, int(c_idxs[3])])
    ax2.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_gd[:, int(c_idxs[3])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax2.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])), marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax2.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax2.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax2.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    legend_used.append(int(c_idxs[3]))
    ax2.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q1)".format(int(c_idxs[3])))

    # Shrink current axis by 20%
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])

    fig2.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q1), cmap=cm1),
                  ax=ax2, orientation='vertical',
                  label='Probability of Category {} (Q1)'.format(int(c_idxs[3])))

    ax2.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax2.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/gd/seed_{}/seed_{}_quad_1.png'.format(save_path, seed, seed))
    plt.close(fig2)

    # Quadrant 2 - Trained TF
    max_q2 = max(max(preds_tf[:, int(c_idxs[1])]), max(preds_gd[:, int(c_idxs[1])]))

    fig3 = plt.figure()
    fig3.set_size_inches(8, 4.8)
    ax3 = plt.subplot(111)

    legend_used = []

    colors_2_tf = cm1(preds_tf[:, int(c_idxs[1])])
    ax3.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_tf[:, int(c_idxs[1])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax3.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])), marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax3.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax3.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax3.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    ax3.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q2)".format(int(c_idxs[1])))

    # Shrink current axis by 20%
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])

    fig3.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q2), cmap=cm1),
                  ax=ax3, orientation='vertical',
                  label='Probability of Category {} (Q2)'.format(int(c_idxs[1])))

    ax3.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax3.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/trained/seed_{}/seed_{}_quad_2.png'.format(save_path, seed, seed))
    plt.close(fig3)

    # Quadrant 2 - GD
    fig4 = plt.figure()
    fig4.set_size_inches(8, 4.8)
    ax4 = plt.subplot(111)

    legend_used = []

    colors_2_gd = cm1(preds_gd[:, int(c_idxs[1])])
    ax4.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_gd[:, int(c_idxs[1])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax4.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])) if int(c_idxs[0]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax4.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax4.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax4.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[4]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    ax4.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q2)".format(int(c_idxs[1])))

    # Shrink current axis by 20%
    box4 = ax4.get_position()
    ax4.set_position([box4.x0, box4.y0, box4.width * 0.8, box4.height])

    fig4.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q2), cmap=cm1),
                  ax=ax4, orientation='vertical',
                  label='Probability of Category {} (Q2)'.format(int(c_idxs[1])))

    ax4.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax4.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/gd/seed_{}/seed_{}_quad_2.png'.format(save_path, seed, seed))
    plt.close(fig4)

    # Quadrant 3 - Trained TF
    max_q3 = max(max(preds_tf[:, int(c_idxs[0])]), max(preds_gd[:, int(c_idxs[0])]))

    fig5 = plt.figure()
    fig5.set_size_inches(8, 4.8)
    ax5 = plt.subplot(111)

    legend_used = []

    colors_3_tf = cm1(preds_tf[:, int(c_idxs[0])])
    ax5.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_tf[:, int(c_idxs[0])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax5.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])) if int(c_idxs[0]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax5.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax5.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax5.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    ax5.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q3)".format(int(c_idxs[0])))

    # Shrink current axis by 20%
    box5 = ax5.get_position()
    ax5.set_position([box5.x0, box5.y0, box5.width * 0.8, box5.height])

    fig5.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q3), cmap=cm1),
                  ax=ax5, orientation='vertical',
                  label='Probability of Category {} (Q3)'.format(int(c_idxs[0])))

    ax5.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax5.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/trained/seed_{}/seed_{}_quad_3.png'.format(save_path, seed, seed))
    plt.close(fig5)

    # Quadrant 3 - GD
    fig6 = plt.figure()
    fig6.set_size_inches(8, 4.8)
    ax6 = plt.subplot(111)

    legend_used = []

    colors_3_gd = cm1(preds_gd[:, int(c_idxs[0])])
    ax6.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_gd[:, int(c_idxs[0])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax6.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])) if int(c_idxs[0]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax6.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax6.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax6.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    ax6.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q3)".format(int(c_idxs[0])))

    # Shrink current axis by 20%
    box6 = ax6.get_position()
    ax6.set_position([box6.x0, box6.y0, box6.width * 0.8, box6.height])

    fig6.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q3), cmap=cm1),
                  ax=ax6, orientation='vertical',
                  label='Probability of Category {} (Q3)'.format(int(c_idxs[0])))

    ax6.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax6.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/gd/seed_{}/seed_{}_quad_3.png'.format(save_path, seed, seed))
    plt.close(fig6)

    # Quadrant 4 - Trained TF
    max_q4 = max(max(preds_tf[:, int(c_idxs[2])]), max(preds_gd[:, int(c_idxs[2])]))

    fig7 = plt.figure()
    fig7.set_size_inches(8, 4.8)
    ax7 = plt.subplot(111)

    legend_used = []

    colors_4_tf = cm1(preds_tf[:, int(c_idxs[2])])
    ax7.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_tf[:, int(c_idxs[2])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax7.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])), marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax7.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax7.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax7.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    ax7.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q4)".format(int(c_idxs[2])))

    # Shrink current axis by 20%
    box7 = ax7.get_position()
    ax7.set_position([box7.x0, box7.y0, box7.width * 0.8, box7.height])

    fig7.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q4), cmap=cm1),
                  ax=ax7, orientation='vertical',
                  label='Probability of Category {} (Q4)'.format(int(c_idxs[2])))

    ax7.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax7.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/trained/seed_{}/seed_{}_quad_4.png'.format(save_path, seed, seed))
    plt.close(fig7)

    # Quadrant 4 - GD
    fig8 = plt.figure()
    fig8.set_size_inches(8, 4.8)
    ax8 = plt.subplot(111)

    legend_used = []

    colors_4_gd = cm1(preds_gd[:, int(c_idxs[2])])
    ax8.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_gd[:, int(c_idxs[2])],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax8.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])) if int(c_idxs[0]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax8.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax8.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax8.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    ax8.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (Q4)".format(int(c_idxs[2])))

    # Shrink current axis by 20%
    box8 = ax8.get_position()
    ax8.set_position([box8.x0, box8.y0, box8.width * 0.8, box8.height])

    fig6.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q4), cmap=cm1),
                  ax=ax8, orientation='vertical',
                  label='Probability of Category {} (Q4)'.format(int(c_idxs[2])))

    ax8.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax8.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/gd/seed_{}/seed_{}_quad_4.png'.format(save_path, seed, seed))
    plt.close(fig8)

    # Random c
    for c in range(20):
        if c not in c_idxs:
            cat = c
            break

    # No Quadrant - Trained TF
    max_q5 = max(max(preds_tf[:, cat]), max(preds_gd[:, cat]))

    fig9 = plt.figure()
    fig9.set_size_inches(8, 4.8)
    ax9 = plt.subplot(111)

    legend_used = []

    colors_5_tf = cm1(preds_tf[:, cat])
    ax9.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_tf[:, cat],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax9.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])) if int(c_idxs[0]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax9.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax9.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax9.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                marker="${}$".format(int(c_idxs[3])), color='white')
    ax9.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (No Quadrant))".format(cat))

    # Shrink current axis by 20%
    box9 = ax9.get_position()
    ax9.set_position([box9.x0, box9.y0, box9.width * 0.8, box9.height])

    fig9.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q5), cmap=cm1),
                  ax=ax9, orientation='vertical',
                  label='Probability of Category {} (No Quadrant)'.format(cat))

    ax9.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax9.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/trained/seed_{}/seed_{}_no_quad.png'.format(save_path, seed, seed))
    plt.close(fig9)

    # No Quadrant - GD
    fig10 = plt.figure()
    fig10.set_size_inches(8, 4.8)
    ax10 = plt.subplot(111)

    legend_used = []

    colors_5_gd = cm1(preds_gd[:, cat])
    ax10.scatter(x_query[:, 0], x_query[:, 1], marker='o', c=preds_gd[:, cat],
                cmap=mcol.LinearSegmentedColormap.from_list("ProbabilityCmap", ["r", "b"]))

    ax10.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[0])],
                label='Category {}'.format(int(c_idxs[0])) if int(c_idxs[0]) not in legend_used else None,
                 marker="${}$".format(int(c_idxs[0])), color='white')
    legend_used.append(int(c_idxs[0]))
    ax10.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[1])],
                label='Category {}'.format(int(c_idxs[1])) if int(c_idxs[1]) not in legend_used else None,
                 marker="${}$".format(int(c_idxs[1])), color='white')
    legend_used.append(int(c_idxs[1]))
    ax10.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[2])],
                label='Category {}'.format(int(c_idxs[2])) if int(c_idxs[2]) not in legend_used else None,
                 marker="${}$".format(int(c_idxs[2])), color='white')
    legend_used.append(int(c_idxs[2]))
    ax10.scatter([x[i, 0] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                [x[i, 1] for i in range(c_size) if x_label[i] == int(c_idxs[3])],
                label='Category {}'.format(int(c_idxs[3])) if int(c_idxs[3]) not in legend_used else None,
                 marker="${}$".format(int(c_idxs[3])), color='white')
    ax10.scatter([x[i, 0] for i in range(c_size) if x_label[i] not in c_idxs],
                [x[i, 1] for i in range(c_size) if x_label[i] not in c_idxs],
                label='Other Categories', marker="o", color='white')

    plt.title("Predicted Probability of Category {} (No Quadrant)".format(cat))

    # Shrink current axis by 20%
    box10 = ax10.get_position()
    ax10.set_position([box10.x0, box10.y0, box10.width * 0.8, box10.height])

    fig10.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, max_q5), cmap=cm1),
                  ax=ax10, orientation='vertical',
                  label='Probability of Category {} (No Quadrant)'.format(cat))

    ax10.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))
    for ha in ax10.legend_.legend_handles:
        ha.set_edgecolor("black")

    plt.tight_layout()

    plt.savefig('{}/gd/seed_{}/seed_{}_no_quad.png'.format(save_path, seed, seed))
    plt.close(fig10)


if __name__ == '__main__':
    args = parser.parse_args()

    gen_figs(args.data_path,
             args.save_path,
             args.num_seeds)

