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


def convert_params(file_path, save_path, layers):
    print("Loading params from {}.pickle".format(file_path))

    with open("{}.pickle".format(file_path), 'rb') as f:
        params = pickle.load(f)

    # print(params)

    file = open("{}_layers_{}.txt".format(save_path, layers), "w+")

    for key in params:
        np.set_printoptions(threshold=sys.maxsize)
        content = str(np.array(params[key]['w']).transpose())
        print(key)
        print(content)
        file.write("{}:\n".format(key))
        file.write(content)
        file.write("\n")

    file.close()


def create_heatmap(file_path, save_path, layers):
    print("Loading params from {}.pickle".format(file_path))

    with open("{}.pickle".format(file_path), 'rb') as f:
        params = pickle.load(f)

    print(params)

    for key in params:
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(suppress=True)
        data = np.abs(np.array(params[key]['w']).transpose())

        name = key.split("/")
        if layers > 1:
            param_name = name[-2] + "_" + name[-1]
        else:
            param_name = "layer_0_" + name[-1]

        fig = plt.figure(figsize=(8, 5))

        plt.imshow(data, cmap='autumn', interpolation='nearest')
        plt.xticks(np.arange(0, data.shape[1], 1.0), rotation=90)
        plt.yticks(np.arange(0, data.shape[0], 1.0))
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel('Absolute Value of Parameters', rotation=270)

        plt.title(param_name)

        plt.tight_layout()
        plt.savefig("{}_{}.jpg".format(save_path, param_name), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    layers = [1]
    seeds = [0, 1, 2, 3, 4]
    # for s in seeds:
    #     convert_params(
    #         "../results_high_dim/cats_100_emb_25/layers_2_gd_plus/c_size_500_lam_10/rbf/params_gd_seed_{}".format(2, s),
    #         "../results_high_dim/cats_100_emb_25/layers_2_gd_plus/c_size_500_lam_10/rbf/params_gd_seed_{}".format(s), 2)

    for s in seeds:
        if not os.path.exists("../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_trained_seed_{}".format(s)):
            os.mkdir("../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_trained_seed_{}".format(s))

        if not os.path.exists("../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_gd_seed_{}".format(s)):
            os.mkdir("../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_gd_seed_{}".format(s))

        convert_params("../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_trained_seed_{}".format(s),
                       "../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_trained_seed_{}".format(s), 1)
        # create_heatmap("../results_final/reg_emb_cats_25_d_10/layers_2/c_size_125_final/softmax/params_trained_seed_{}".format(s),
        #                "../results_final/reg_emb_cats_25_d_10/layers_2/c_size_125_final/softmax/params_trained_seed_{}/params_trained_seed_{}".format(s, s), 1)

        convert_params(
            "../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_gd_seed_{}".format(s),
            "../results_final/imagenet_reg_emb_cats_5/layers_1/c_size_50_final_new/softmax/params_gd_seed_{}".format(s), 1)
        # create_heatmap(
        #     "../results_final/reg_emb_cats_25_d_10/layers_2/c_size_125_final/softmax/params_gd_seed_{}".format(s),
        #     "../results_final/reg_emb_cats_25_d_10/layers_2/c_size_125_final/softmax/params_gd_seed_{}/params_gd_seed_{}".format(
        #         s, s), 1)


    # if not os.path.exists("../reg_params"):
    #     os.mkdir("../reg_params")
    #
    # for l in layers:
    #     if not os.path.exists("../reg_params/{}_layer".format(l)):
    #         os.mkdir("../reg_params/{}_layer".format(l))
    #
    #     if not os.path.exists("../reg_params/{}_layer/softmax".format(l)):
    #         os.mkdir("../reg_params/{}_layer/softmax".format(l))
    #
    #     for s in seeds:
    #         if not os.path.exists("../reg_params/{}_layer/softmax/params_trained_seed_{}".format(l, s)):
    #             os.mkdir("../reg_params/{}_layer/softmax/params_trained_seed_{}".format(l, s))
    #
    #         if not os.path.exists("../reg_params/{}_layer/softmax/params_gd_seed_{}".format(l, s)):
    #             os.mkdir("../reg_params/{}_layer/softmax/params_gd_seed_{}".format(l, s))
    #
    #         convert_params("../results/reg_emb/layers_{}/c_size_125_v3/softmax/params_trained_seed_{}".format(l, s),
    #                        "../reg_params/{}_layer/softmax/params_trained_seed_{}".format(l, s), l)
    #         create_heatmap("../results/reg_emb/layers_{}/c_size_125_v3/softmax/params_trained_seed_{}".format(l, s),
    #                        "../reg_params/{}_layer/softmax/params_trained_seed_{}/params_trained_seed_{}".format(
    #                            l, s, s), l)
    #
    #         convert_params("../results/reg_emb/layers_{}/c_size_125_v3/softmax/params_gd_seed_{}".format(l, s),
    #                        "../reg_params/{}_layer/softmax/params_gd_seed_{}".format(l, s), l)
    #         create_heatmap("../results/reg_emb/layers_{}/c_size_125_v3/softmax/params_gd_seed_{}".format(l, s),
    #                        "../reg_params/{}_layer/softmax/params_gd_seed_{}/params_gd_seed_{}".format(l, s, s), l)
    #
    # if not os.path.exists("../one_hot_params"):
    #     os.mkdir("../one_hot_params")
    #
    # for l in layers:
    #     if not os.path.exists("../one_hot_params/{}_layer".format(l)):
    #         os.mkdir("../one_hot_params/{}_layer".format(l))
    #
    #     if not os.path.exists("../one_hot_params/{}_layer/softmax".format(l)):
    #         os.mkdir("../one_hot_params/{}_layer/softmax".format(l))
    #
    #     for s in seeds:
    #         if not os.path.exists("../one_hot_params/{}_layer/softmax/params_trained_seed_{}".format(l, s)):
    #             os.mkdir("../one_hot_params/{}_layer/softmax/params_trained_seed_{}".format(l, s))
    #
    #         if not os.path.exists("../one_hot_params/{}_layer/softmax/params_gd_seed_{}".format(l, s)):
    #             os.mkdir("../one_hot_params/{}_layer/softmax/params_gd_seed_{}".format(l, s))
    #
    #         convert_params("../results/one_hot_emb/layers_{}/c_size_125_v1/softmax/params_trained_seed_{}".format(l, s),
    #                        "../one_hot_params/{}_layer/softmax/params_trained_seed_{}".format(l, s), l)
    #         create_heatmap("../results/one_hot_emb/layers_{}/c_size_125_v1/softmax/params_trained_seed_{}".format(l, s),
    #                        "../one_hot_params/{}_layer/softmax/params_trained_seed_{}/params_trained_seed_{}".format(
    #                            l, s, s), l)
    #
    #         convert_params("../results/one_hot_emb/layers_{}/c_size_125_v1/softmax/params_gd_seed_{}".format(l, s),
    #                        "../one_hot_params/{}_layer/softmax/params_gd_seed_{}".format(l, s), l)
    #         create_heatmap("../results/one_hot_emb/layers_{}/c_size_125_v1/softmax/params_gd_seed_{}".format(l, s),
    #                        "../one_hot_params/{}_layer/softmax/params_gd_seed_{}/params_gd_seed_{}".format(l, s, s), l)
