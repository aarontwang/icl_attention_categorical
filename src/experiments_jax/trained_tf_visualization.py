import os
import sys
import pickle
import json

import math
import numpy as np

import matplotlib.pylab as pl
import matplotlib.pyplot as plt

colors = pl.colormaps['Dark2']

pl.rcParams.update({'font.size': 12})
pl.rc('axes', labelsize=14)
pl.rcParams.update({
    "text.usetex": False,
})


def compute_value_proj(file_path, save_path, title, i_size, e_size):
    if not os.path.exists("{}/E_D".format(save_path)):
        os.mkdir("{}/E_D".format(save_path))

    if not os.path.exists("{}/Q".format(save_path)):
        os.mkdir("{}/Q".format(save_path))

    print("Loading params from {}.pickle".format(file_path))

    with open("{}.pickle".format(file_path), 'rb') as f:
        params = pickle.load(f)

    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    value = np.array(params['transformer/multi_head_attention/value']['w']).transpose()
    projection = np.array(params['transformer/multi_head_attention/linear']['w']).transpose()

    D = value[:, i_size:(i_size + e_size)]
    E = projection[-e_size:, :]

    gamma = E @ D

    file = open("{}/E_D/{}_E_D.txt".format(save_path, title), "w+")
    file.write("E * D\n")
    file.write(str(gamma))
    file.write("\n")
    file.close()

    fig = plt.figure(figsize=(8, 5))

    plt.imshow(np.abs(gamma), cmap='autumn', interpolation='nearest')
    plt.xticks(np.arange(0, gamma.shape[1], 1.0), rotation=90)
    plt.yticks(np.arange(0, gamma.shape[0], 1.0))
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 25
    cbar.ax.set_ylabel('Absolute Value of ED', rotation=270)

    plt.title("ED")

    plt.tight_layout()
    plt.savefig("{}/E_D/{}_E_D.jpg".format(save_path, title))
    plt.close()

    query = np.array(params['transformer/multi_head_attention/query']['w']).transpose()
    key = np.array(params['transformer/multi_head_attention/key']['w']).transpose()

    A = query[:, :i_size]
    B = key[:, :i_size]

    Q = A.T @ B

    file = open("{}/Q/{}_Q.txt".format(save_path, title), "w+")
    file.write("Q\n")
    file.write(str(Q))
    file.write("\n")
    file.close()

    fig = plt.figure(figsize=(8, 5))

    plt.imshow(np.abs(Q), cmap='autumn', interpolation='nearest')
    plt.xticks(np.arange(0, Q.shape[1], 1.0), rotation=90)
    plt.yticks(np.arange(0, Q.shape[0], 1.0))
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 25
    cbar.ax.set_ylabel('Absolute Value of Q', rotation=270)

    plt.title("Q")

    plt.tight_layout()
    plt.savefig("{}/Q/{}_Q.jpg".format(save_path, title))
    plt.close()


if __name__ == '__main__':
    layers = [1]
    seeds = [0]

    for l in layers:
        for s in seeds:
            if not os.path.exists("../grid_params_visualization/softmax"):
                os.mkdir("../grid_params_visualization/softmax")

            compute_value_proj("../results/grid_c_size_20/layers_{}/softmax/params_trained_seed_{}".format(l, s),
                               "../grid_params_visualization/softmax", "params_trained_seed_{}".format(s), 2, 2)
