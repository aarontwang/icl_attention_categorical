import sys
import os

import matplotlib.pylab as pl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import json

colors = pl.colormaps['Dark2']

pl.rcParams.update({'font.size': 12})
pl.rc('axes', labelsize=14)
pl.rcParams.update({
    "text.usetex": False,
})


def plot_comparison(save_path, linear, exp, rbf, lap, softmax, max_prob=None,
                    num_iter_os=None, plot_title=None, title=None,
                    x_label='Training Epochs', y_label='Loss',
                    yscale_log=False, y_lim_l=0, y_lim_u=1,
                    color_add=0, loc_first='upper right', width=4, height=3.5, use_legend=True):
    print("Generating plot: ", title)

    if linear is not None:
        linear = np.array(linear)
    if exp is not None:
        exp = np.array(exp)
    if rbf is not None:
        rbf = np.array(rbf)
    if lap is not None:
        lap = np.array(lap)
    if softmax is not None:
        softmax = np.array(softmax)

    if linear is not None:
        linear_std = np.std(linear, axis=0)
        linear = np.mean(linear, axis=0)

    if exp is not None:
        exp_std = np.std(exp, axis=0)
        exp = np.mean(exp, axis=0)

    if rbf is not None:
        rbf_std = np.std(rbf, axis=0)
        rbf = np.mean(rbf, axis=0)

    if lap is not None:
        lap_std = np.std(lap, axis=0)
        lap = np.mean(lap, axis=0)

    if softmax is not None:
        softmax_std = np.std(softmax, axis=0)
        softmax = np.mean(softmax, axis=0)

    if max_prob is not None:
        max_prob = np.mean(max_prob, axis=0)

    # create figure and axes
    fig, ax1 = pl.subplots()
    ax1.set_xlabel(x_label)
    fig.set_size_inches(width, height)

    # Training Epochs
    x_range = np.arange(0, num_iter_os)

    # linear
    if linear is not None:
        ax1.plot(x_range, linear, color=colors(0.1 + color_add), label='Linear', linewidth='3')
        ax1.fill_between(x_range, linear - linear_std, linear + linear_std,
                         alpha=0.2, facecolor=colors(0.1 + color_add))

    # exp
    if exp is not None:
        ax1.plot(x_range, exp, color=colors(0.2 + color_add), label='Exp', linewidth='3')
        ax1.fill_between(x_range, exp - exp_std, exp + exp_std,
                         alpha=0.2, facecolor=colors(0.2 + color_add))

    if rbf is not None:
        # rbf
        ax1.plot(x_range, rbf, color=colors(0.3 + color_add), label='RBF', linewidth='3')
        ax1.fill_between(x_range, rbf - rbf_std, rbf + rbf_std,
                         alpha=0.2, facecolor=colors(0.3 + color_add))

    if softmax is not None:
        # softmax
        ax1.plot(x_range, softmax, color=colors(0.4 + color_add), label='Softmax', linewidth='3')
        ax1.fill_between(x_range, softmax - softmax_std, softmax + softmax_std,
                         alpha=0.5, facecolor=colors(0.4 + color_add))

    if lap is not None:
        # laplacian
        ax1.plot(x_range, lap, color=colors(0.5 + color_add), label='Laplacian', linewidth='3')
        ax1.fill_between(x_range, lap - lap_std, lap + lap_std,
                         alpha=0.2, facecolor=colors(0.5 + color_add))

    if max_prob is not None:
        ax1.plot(x_range, max_prob, color='k', label='Mean Max Probability', linewidth='3')

    # Put a legend to the right of the current axis
    # if use_legend:
    #     box = ax1.get_position()
    #     ax1.set_position([box.x0, box.y0, box.width * 0.5, box.height])
    #     ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.99, facecolor='white')
    #     # legend1 = ax1.legend(loc=loc_first, framealpha=0.99, facecolor='white')
    #     # legend1.set_zorder(100)

    ax1.legend(loc='best', framealpha=1, facecolor='white')
    ax1.spines['right'].set_visible(False)

    legend1 = ax1.legend(loc=loc_first, framealpha=0.99, facecolor='white')
    legend1.set_zorder(100)

    ax1.spines['right'].set_visible(False)

    plt.ylabel(y_label)
    plt.ylim(y_lim_l, y_lim_u)

    ax1.spines['top'].set_visible(False)

    if plot_title is not None:
        plt.title(plot_title)

    if yscale_log:
        ax1.set_yscale("log")

    plt.tight_layout()

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plt.savefig(save_path + "/" + title + ".png", format="png", bbox_inches="tight", pad_inches=0)

    plt.close()

    print("Done generating plot")


def compare_kernels(save_path, linear=False, exp=False, rbf=False, laplacian=False,
                       softmax=False, train=False, imagenet=False):
    kernels = ['linear', 'exp', 'rbf', 'laplacian', 'softmax']

    if linear:
        linear_loss_trans_list, linear_losses_gd_list, linear_acc_trans_list, linear_acc_gd_list, \
            linear_max_prob = extract_data(save_path, 'linear', False, 5000, train)

    if rbf:
        rbf_loss_trans_list, rbf_losses_gd_list, \
            rbf_acc_trans_list, rbf_acc_gd_list, \
            rbf_max_prob = extract_data(save_path, 'rbf', False, 5000, train)

    if softmax:
        softmax_loss_trans_list, softmax_losses_gd_list, \
            softmax_acc_trans_list, softmax_acc_gd_list, \
            softmax_max_prob = extract_data(save_path, 'softmax', False, 5000, train)

    plot_comparison(save_path, linear_loss_trans_list if linear else None, None, rbf_loss_trans_list if rbf else None, None, softmax_loss_trans_list,
                    max_prob=None,
                    num_iter_os=5000, plot_title=None, title="trained_tf_loss",
                    x_label='Training Epochs', y_label='Negative Log-Likelihood Loss',
                    yscale_log=False, y_lim_l=0, y_lim_u=3.5,
                    color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)

    plot_comparison(save_path, linear_losses_gd_list if linear else None, None, rbf_losses_gd_list if rbf else None, None, softmax_losses_gd_list,
                    max_prob=None,
                    num_iter_os=5000, plot_title=None,
                    title="gd_train_loss" if train else "gd_loss",
                    x_label='Training Epochs', y_label='Negative Log-Likelihood Loss',
                    yscale_log=False, y_lim_l=0, y_lim_u=3.5,
                    color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)

    # if not imagenet:
    #     plot_comparison(save_path, linear_prob_loss_trans_list if linear else None, None, rbf_prob_loss_trans_list if rbf else None, None, softmax_prob_loss_trans_list,
    #                     max_prob=None,
    #                     num_iter_os=5000, plot_title=None, title="trained_tf_mse_prob",
    #                     x_label='Training Epochs', y_label='MSE on Category Probabilities',
    #                     yscale_log=False, y_lim_l=0, y_lim_u=0.3,
    #                     color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)
    #
    #     plot_comparison(save_path, linear_prob_loss_gd_list if linear else None, None, rbf_prob_loss_gd_list if rbf else None, None, softmax_prob_loss_gd_list,
    #                     max_prob=None,
    #                     num_iter_os=5000, plot_title=None,
    #                     title="gd_train_mse_prob" if train else "gd_mse_prob",
    #                     x_label='Training Epochs', y_label='MSE on Category Probabilities',
    #                     yscale_log=False, y_lim_l=0, y_lim_u=0.3,
    #                     color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)

    plot_comparison(save_path, linear_acc_trans_list if linear else None, None, rbf_acc_trans_list if rbf else None, None, softmax_acc_trans_list,
                    max_prob=softmax_max_prob,
                    num_iter_os=5000, plot_title=None, title="trained_tf_acc",
                    x_label='Training Epochs', y_label='Accuracy',
                    yscale_log=False, y_lim_l=0, y_lim_u=1,
                    color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)

    plot_comparison(save_path, linear_acc_gd_list if linear else None, None, rbf_acc_gd_list if rbf else None, None, softmax_acc_gd_list,
                    max_prob=softmax_max_prob,
                    num_iter_os=5000, plot_title=None, title="gd_train_acc" if train else "gd_acc",
                    x_label='Training Epochs', y_label='Negative Log-Likelihood Loss',
                    yscale_log=False, y_lim_l=0, y_lim_u=1,
                    color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)

    # plot_comparison(save_path, linear_top_3_freq_trans_list if linear else None, None, rbf_top_3_freq_trans_list if rbf else None, None, softmax_top_3_freq_trans_list,
    #                 max_prob=None,
    #                 num_iter_os=5000, plot_title=None, title="trained_tf_top_3_freq",
    #                 x_label='Training Epochs', y_label='Top 3 Frequency',
    #                 yscale_log=False, y_lim_l=0, y_lim_u=1,
    #                 color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)
    #
    # plot_comparison(save_path, linear_top_3_freq_gd_list if linear else None, None, rbf_top_3_freq_gd_list if rbf else None, None, softmax_top_3_freq_gd_list,
    #                 max_prob=None,
    #                 num_iter_os=5000, plot_title=None,
    #                 title="gd_train_top_3_freq" if train else "gd_top_3_freq",
    #                 x_label='Training Epochs', y_label='Top 3 Frequency',
    #                 yscale_log=False, y_lim_l=0, y_lim_u=1,
    #                 color_add=0, loc_first='best', width=6, height=4.5, use_legend=True)


def compare_layers(save_path, data_path_1_ff, data_path_1_cross, data_path_2_ff, data_path_2_cross, data_path_3_ff, data_path_3_cross, data_path_4_ff, data_path_4_cross, early_stopping=False):
    data_names = [('tf_eval_loss_list', 'gd_eval_loss_list'), ('tf_eval_acc_list', 'gd_eval_acc_list')]
    plot_titles = ["Negative Log-Likelihood Loss", "Accuracy", "Top 3 Frequency", "MSE on Category Probabilities"]
    save_titles = ["loss_layer_comp", "acc_layer_comp", "top_3_freq_layer_comp", "mse_prob_layer_comp"]
    y_lim_l = [0.9, 0.2, 0, 0]
    y_lim_u = [3, 0.7, 1, 0.3]

    y_lim_l_tf = [0.9, 0.2, 0, 0]
    y_lim_u_tf = [3.0, 0.7, 1, 0.3]

    layers = [1, 2, 3, 4]

    kernels = ['softmax']
    labels = ["Softmax"]

    for j, kernel in enumerate(kernels):
        for m, (data_tf, data_gd) in enumerate(data_names):
            print(data_gd)
            plot_comparison_multiple_layers([[data_path_1_ff, data_path_1_cross], [data_path_2_ff, data_path_2_cross], [data_path_3_ff, data_path_3_cross], [data_path_4_ff, data_path_4_cross]],
                                            save_path, data_tf, data_gd, layers, kernel,
                                            labels[j],
                                            num_iter_os_1=5000, num_iter_os_2=5000,
                                            plot_title=None, title=save_titles[m], x_label="Training Epochs",
                                            y_label=plot_titles[m], yscale_log=False, y_lim_l=y_lim_l[m],
                                            y_lim_u=y_lim_u[m], y_lim_l_tf=y_lim_l_tf[m], y_lim_u_tf=y_lim_u_tf[m],
                                            color_add=0, loc_first='best', width=6, height=4.5,
                                            early_stopping=early_stopping)


def plot_comparison_multiple_layers(data_paths, save_path, data_tf, data_gd, layers, kernel, kernel_label,
                                    num_iter_os_1=None, num_iter_os_2=None, plot_title=None, title=None,
                                    x_label='Training Epochs', y_label='Loss',
                                    yscale_log=False, y_lim_l=0, y_lim_u=1, y_lim_l_tf=0, y_lim_u_tf=1,
                                    color_add=0, loc_first='best', width=4, height=3.5, use_legend=True,
                                    early_stopping=False):
    # create figure and axes
    fig1, ax1 = pl.subplots()
    ax1.set_xlabel(x_label)
    fig1.set_size_inches(width, height)

    fig2, ax2 = pl.subplots()
    ax2.set_xlabel(x_label)
    fig2.set_size_inches(width, height)

    steps = 5000

    color_vals = [0.05, 0.15, 0.25, 0.65, 0.45, 0.55]
    linestyle = ['-', '--']

    for layer in layers:
        for k in range(2):
            data_f = open(data_paths[layer-1][k] + '/{}/results.json'.format(kernel))
            print(data_paths[layer-1][k] + '/{}/results.json'.format(kernel))
            data = json.load(data_f)
            print(data_gd)

            trans_list = np.array(list(filter(None, data[data_tf])))
            gd_list = np.array(list(filter(None, data[data_gd])))

            if early_stopping:
                best_idx_trans_list = data['tf_best_step_list']
                best_idx_gd_list = data['gd_best_step_list']
                for i, idx in enumerate(best_idx_trans_list):
                    trans_list[i][idx + 1:] = [trans_list[i][idx]] * (steps - idx - 1)
                
                for i, idx in enumerate(best_idx_gd_list):
                    gd_list[i][idx + 1:] = [gd_list[i][idx]] * (steps - idx - 1)
            
            trans_list_mean = np.mean(trans_list, axis=0)
            gd_list_mean = np.mean(gd_list, axis=0)

            trans_list_std = np.std(trans_list, axis=0)
            gd_list_std = np.std(gd_list, axis=0)

            # Epochs
            x_range_1 = np.arange(0, num_iter_os_1)
            x_range_2 = np.arange(0, num_iter_os_2)

            if k==0:
                ax1.plot(x_range_1, trans_list_mean, color=colors(color_vals[layer-1] + color_add),
                        label="{} Layer - FF".format(layer),
                        linewidth='3', linestyle=linestyle[k]) 
            else:
                ax1.plot(x_range_1, trans_list_mean, color=colors(color_vals[layer-1] + color_add),
                        label="{} Layer - Cross-Attention".format(layer),
                        linewidth='3', linestyle=linestyle[k])
            ax1.fill_between(x_range_1, trans_list_mean - trans_list_std, trans_list_mean + trans_list_std,
                            alpha=0.2, facecolor=colors(color_vals[layer-1] + color_add))

            if k==0:
                ax2.plot(x_range_2, gd_list_mean, color=colors(color_vals[layer-1] + color_add), label="{} Layer - FF".format(layer),
                    linewidth='3', linestyle=linestyle[k])
            else:
                ax2.plot(x_range_2, gd_list_mean, color=colors(color_vals[layer-1] + color_add), label="{} Layer - Cross-Attention".format(layer),
                    linewidth='3', linestyle=linestyle[k])
            ax2.fill_between(x_range_2, gd_list_mean - gd_list_std, gd_list_mean + gd_list_std,
                            alpha=0.2, facecolor=colors(color_vals[layer-1] + color_add))

    # if 'acc' in data_tf:
    #     max_prob = np.mean(np.array(data['max_prob_list']), axis=0)
     #
    #     ax1.plot(x_range_1, max_prob, color='k', label='Mean Max Probability', linewidth='3')
    #
    #     ax2.plot(x_range_2, max_prob, color='k', label='Mean Max Probability', linewidth='3')

    ax1.legend(loc='best', framealpha=1, facecolor='white')
    ax1.spines['right'].set_visible(False)
    ax2.legend(loc='best', framealpha=1, facecolor='white')
    ax2.spines['right'].set_visible(False)

    legend1 = ax1.legend(loc=loc_first, framealpha=0.99, facecolor='white', ncol=1)
    legend1.set_zorder(100)
    legend2 = ax2.legend(loc=loc_first, framealpha=0.99, facecolor='white', ncol=1)
    legend2.set_zorder(100)

    ax1.set_ylabel(y_label)
    ax1.set_ylim(y_lim_l_tf, y_lim_u_tf)
    ax1.spines['top'].set_visible(False)

    ax2.set_ylabel(y_label)
    ax2.set_ylim(y_lim_l, y_lim_u)
    ax2.spines['top'].set_visible(False)

    if plot_title is not None:
        ax1.set_title(kernel_label + " " + plot_title)
        ax2.set_title(kernel_label + " " + plot_title)

    if yscale_log:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    fig1.tight_layout()
    fig2.tight_layout()

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if early_stopping:
        fig1.savefig(save_path + "/{}_trained_tf_".format(kernel) + title + "_es.jpg", format="jpg", bbox_inches="tight",
                        pad_inches=0)
        fig2.savefig(save_path + "/{}_gd_".format(kernel) + title + "_es.jpg", format="jpg", bbox_inches="tight",
                        pad_inches=0)
        print(save_path + "/{}_trained_tf_".format(kernel) + title + "_es.jpg")
        print(save_path + "/{}_gd_".format(kernel) + title + "_es.jpg")

    else:
        fig1.savefig(save_path + "/{}_trained_tf_".format(kernel) + title + ".png", format="png", bbox_inches="tight", pad_inches=0)
        fig2.savefig(save_path + "/{}_gd_".format(kernel) + title + ".png", format="png", bbox_inches="tight", pad_inches=0)

    plt.close()

    print("Done generating plot")    


def context_comparison(save_path, linear=False, exp=False, rbf=False, laplacian=False,
                       softmax=False, imagenet=False, cats=25):
    data_names = [('tf_eval_loss_list', 'loss_gd_list'), ('acc_trans_list', 'acc_gd_list')]
    plot_titles = ["Negative Log-Likelihood Loss", "Accuracy", "Top 3 Frequency", "MSE on Category Probabilities"]
    save_titles = ["loss_context_comp", "acc_context_comp", "top_3_freq_context_comp", "mse_prob_context_comp"]
    y_lim_l = [0, 0, 0, 0]
    y_lim_u = [3.5, 1, 1, 0.3]
    locs = ['lower right', 'lower right', 'lower right', 'upper right']

    for i, (data_tf, data_gd) in enumerate(data_names):
        if not (imagenet and 'prob' in data_tf):
            print(data_tf)
            plot_c_size_comparison(save_path, data_tf, data_gd, cats, linear, exp, rbf, laplacian, softmax,
                                   plot_title=None, title=save_titles[i], x_label="Context Size (N)",
                                   y_label=plot_titles[i], yscale_log=False, y_lim_l=y_lim_l[i], y_lim_u=y_lim_u[i],
                                   color_add=0, loc_first=locs[i], width=6, height=4.5)


def plot_c_size_comparison(save_path, data_tf, data_gd, cats, linear=False, exp=False, rbf=False, laplacian=False,
                           softmax=False, plot_title=None, title=None,
                           x_label='Context Size', y_label='Final Loss',
                           yscale_log=False, y_lim_l=0, y_lim_u=1,
                           color_add=0, loc_first='lower left', width=4, height=3.5):
    print("Generating plot: ", title)

    kernels = ['linear', 'exp', 'rbf', 'laplacian', 'softmax']
    labels = ['Linear', 'Exp', 'RBF', 'Laplacian', 'Softmax']

    # create figure and axes
    fig1, ax1 = pl.subplots()
    ax1.set_xlabel(x_label)
    fig1.set_size_inches(width, height)

    fig2, ax2 = pl.subplots()
    ax2.set_xlabel(x_label)
    fig2.set_size_inches(width, height)

    color_vals = {'linear': 0.1, 'rbf': 0.3, 'softmax': 0.4, 'linear_no_scale': 0.2, 'rbf_no_scale': 0.7}

    for i, kernel in enumerate(kernels):
        if (not linear and kernel == 'linear') or (not exp and kernel == 'exp') or \
                (not rbf and kernel == 'rbf') or (not laplacian and kernel == 'laplacian') or \
                (not softmax and kernel == 'softmax'):
            continue

        data_f = open(save_path + '/{}/context_results_scale.json'.format(kernel))
        data = json.load(data_f)

        trans_list = np.array(data[data_tf])
        gd_list = np.array(data[data_gd])

        trans_list_mean = np.mean(trans_list, axis=0)
        gd_list_mean = np.mean(gd_list, axis=0)

        trans_list_std = np.std(trans_list, axis=0)
        gd_list_std = np.std(gd_list, axis=0)

        if kernel != 'softmax':
            data_f_unscaled = open(save_path + '/{}/context_results_no_scale.json'.format(kernel))
            data_unscaled = json.load(data_f_unscaled)

            trans_list_unscaled = np.array(data_unscaled[data_tf])
            gd_list_unscaled = np.array(data_unscaled[data_gd])

            trans_list_unscaled_mean = np.mean(trans_list_unscaled, axis=0)
            gd_list_unscaled_mean = np.mean(gd_list_unscaled, axis=0)

            trans_list_unscaled_std = np.std(trans_list_unscaled, axis=0)
            gd_list_unscaled_std = np.std(gd_list_unscaled, axis=0)

        # Training Epochs
        x_range = cats * np.array(data['factors'])

        ax1.plot(x_range, trans_list_mean, color=colors(color_vals[kernel]), label=labels[i], linewidth='3')
        ax1.fill_between(x_range, trans_list_mean - trans_list_std, trans_list_mean + trans_list_std,
                         alpha=0.2, facecolor=colors(color_vals[kernel]))

        ax2.plot(x_range, gd_list_mean, color=colors(color_vals[kernel]), label=labels[i], linewidth='3')
        ax2.fill_between(x_range, gd_list_mean - gd_list_std, gd_list_mean + gd_list_std,
                         alpha=0.2, facecolor=colors(color_vals[kernel]))

        if kernel != 'softmax':
            ax1.plot(x_range, trans_list_unscaled_mean, color=colors(color_vals[kernel + '_no_scale']),
                     label=labels[i] + " No Scaling", linewidth='3')
            ax1.fill_between(x_range, trans_list_unscaled_mean - trans_list_unscaled_std,
                             trans_list_unscaled_mean + trans_list_unscaled_std,
                             alpha=0.2, facecolor=colors(color_vals[kernel + '_no_scale']))

            ax2.plot(x_range, gd_list_unscaled_mean, color=colors(color_vals[kernel + '_no_scale']),
                     label=labels[i] + " No Scaling", linewidth='3')
            ax2.fill_between(x_range, gd_list_unscaled_mean - gd_list_unscaled_std,
                             gd_list_unscaled_mean + gd_list_unscaled_std,
                             alpha=0.2, facecolor=colors(color_vals[kernel + '_no_scale']))

    ax1.set_xticks(x_range)
    ax2.set_xticks(x_range)

    ax1.legend(loc=loc_first, framealpha=1, facecolor='white', fontsize=10, ncol=5)
    ax1.spines['right'].set_visible(False)
    ax2.legend(loc=loc_first, framealpha=1, facecolor='white', fontsize=10, ncol=5)
    ax2.spines['right'].set_visible(False)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    order = [0, 2, 4, 1, 3]

    legend1 = ax1.legend([handles1[idx] for idx in order], [labels1[idx] for idx in order],
                         loc=loc_first, framealpha=0.99, facecolor='white', fontsize=10, ncol=2)
    legend1.set_zorder(100)
    legend2 = ax2.legend([handles2[idx] for idx in order], [labels2[idx] for idx in order],
                         loc=loc_first, framealpha=0.99, facecolor='white', fontsize=10, ncol=2)
    legend2.set_zorder(100)

    ax1.set_ylabel(y_label)
    ax1.set_ylim(y_lim_l, y_lim_u)
    ax1.spines['top'].set_visible(False)

    ax2.set_ylabel(y_label)
    ax2.set_ylim(y_lim_l, y_lim_u)
    ax2.spines['top'].set_visible(False)

    if plot_title is not None:
        ax1.set_title("Trained TF " + plot_title)
        ax2.set_title("GD " + plot_title)

    if yscale_log:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    fig1.tight_layout()
    fig2.tight_layout()

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    fig1.savefig(save_path + "/trained_tf_" + title + ".png", format="png", bbox_inches="tight", pad_inches=0)
    fig2.savefig(save_path + "/gd_" + title + ".png", format="png", bbox_inches="tight", pad_inches=0)

    plt.close()

    print("Done generating plot")


def extract_data(data_path, model_kernel, early_stopping=False, steps=5000, train=False):
    data_f = open(data_path + '/{}/results.json'.format(model_kernel))
    data = json.load(data_f)

    loss_trans_list = list(filter(None, data['tf_eval_loss_list']))
    losses_gd_list = list(filter(None, data['gd_eval_loss_list'] if train else data['tf_gd_eval_loss_list']))
    # prob_loss_trans_list = list(filter(None, data['prob_dist_trans_list']))
    # prob_loss_gd_list = list(filter(None, data['gd_train_prob_dist_list'] if train else data['prob_dist_gd_list']))
    acc_trans_list = list(filter(None, data['tf_eval_acc_list']))
    acc_gd_list = list(filter(None, data['gd_eval_acc_list'] if train else data['tf_gd_eval_acc_list']))
    # top_3_freq_trans_list = list(filter(None, data['top_3_freq_trans_list']))
    # top_3_freq_gd_list = list(filter(None, data['gd_train_top_3_freq_list'] if train else data['top_3_freq_gd_list']))
    max_prob = None

    if early_stopping:
        best_idx_trans_list = data['tf_best_step_list']
        best_idx_gd_list = data['gd_best_step_list']
        for i, idx in enumerate(best_idx_trans_list):
            gd_idx = best_idx_gd_list[i]
            if len(loss_trans_list[i]) > 0:
                loss_trans_list[i][idx + 1:] = [loss_trans_list[i][idx]] * (steps - idx - 1)
                losses_gd_list[i][gd_idx + 1:] = [losses_gd_list[i][gd_idx]] * (steps - gd_idx - 1)

                # prob_loss_trans_list[i][idx + 1:] = [prob_loss_trans_list[i][idx]] * (
                #         steps - idx - 1)
                # prob_loss_gd_list[i][gd_idx + 1:] = [prob_loss_gd_list[i][gd_idx]] * (steps - gd_idx - 1)

                acc_trans_list[i][idx + 1:] = [acc_trans_list[i][idx]] * (steps - idx - 1)
                acc_gd_list[i][gd_idx + 1:] = [acc_gd_list[i][gd_idx]] * (steps - gd_idx - 1)
                #
                # top_3_freq_trans_list[i][idx + 1:] = [top_3_freq_trans_list[i][idx]] * (steps - idx - 1)
                # top_3_freq_gd_list[i][gd_idx + 1:] = [top_3_freq_gd_list[i][gd_idx]] * (steps - gd_idx - 1)

                # max_prob[i][idx + 1:] = [max_prob[i][-1]] * (steps - idx - 1)

    return loss_trans_list, losses_gd_list, acc_trans_list, acc_gd_list, max_prob


def plot_training(save_path, data_path, early_stopping=True, linear=False, exp=False, rbf=False,
                  laplacian=False, softmax=False, gd_plus=False, imagenet=False, iterations=5000):
    kernels = ['linear', 'exp', 'rbf', 'laplacian', 'softmax']
    labels = ["Linear", "Exp", "RBF", "Laplacian", "Softmax"]

    steps = iterations

    for j, model_kernel in enumerate(kernels):
        if (not linear and model_kernel == 'linear') or (not exp and model_kernel == 'exp') or \
                (not rbf and model_kernel == 'rbf') or (not laplacian and model_kernel == 'laplacian') or \
                (not softmax and model_kernel == 'softmax'):
            continue

        data_f = open(data_path + '/{}/results.json'.format(model_kernel))
        data = json.load(data_f)

        loss_trans_list = list(filter(None, data['tf_eval_loss_list']))
        losses_gd_list = list(filter(None, data['tf_gd_eval_loss_list']))
        # if not imagenet:
        #     prob_loss_trans_list = list(filter(None, data['prob_dist_trans_list']))
        #     prob_loss_gd_list = list(filter(None, data['prob_dist_gd_list']))
        acc_trans_list = list(filter(None, data['tf_eval_acc_list']))
        acc_gd_list = list(filter(None, data['tf_gd_eval_acc_list']))
        # top_3_freq_trans_list = list(filter(None, data['top_3_freq_trans_list']))
        # top_3_freq_gd_list = list(filter(None, data['top_3_freq_gd_list']))
        # max_prob = data['max_prob_list']

        # print(max_prob)

        gd_train_loss = list(filter(None, data['gd_eval_loss_list']))
        # if not imagenet:
        #     gd_train_prob_dist = list(filter(None, data['gd_train_prob_dist_list']))
        gd_train_acc = list(filter(None, data['gd_eval_acc_list']))
        # gd_train_top_3_freq = list(filter(None, data['gd_train_top_3_freq_list']))

        if early_stopping:
            best_idx_trans_list = data['tf_best_step_list']
            for i, idx in enumerate(best_idx_trans_list):
                loss_trans_list[i][idx + 1:] = [loss_trans_list[i][idx]] * (steps - idx - 1)

                # if not imagenet:
                #     prob_loss_trans_list[i][idx + 1:] = [prob_loss_trans_list[i][idx]] * (
                #                 steps - idx - 1)

                acc_trans_list[i][idx + 1:] = [acc_trans_list[i][idx]] * (steps - idx - 1)

                # top_3_freq_trans_list[i][idx + 1:] = [top_3_freq_trans_list[i][idx]] * (steps - idx - 1)

            best_idx_gd_list = data['gd_best_step_list']
            for i, idx in enumerate(best_idx_gd_list):
                gd_train_loss[i][idx + 1:] = [gd_train_loss[i][idx]] * (steps - idx - 1)

                # if not imagenet:
                #     gd_train_prob_dist[i][idx + 1:] = [gd_train_prob_dist[i][idx]] * (
                #                 steps - idx - 1)

                gd_train_acc[i][idx + 1:] = [gd_train_acc[i][idx]] * (steps - idx - 1)

               #  gd_train_top_3_freq[i][idx + 1:] = [gd_train_top_3_freq[i][idx]] * (steps - idx -1)

        # best_step = int(data['gd_val_best_step']/100)

        # for i in range(len(gd_train_loss)):
        #     gd_train_loss[i] = gd_train_loss[i] + [gd_train_loss[i][best_step]] * (len(loss_trans_list[i]) - len(gd_train_loss[i]))
        #     gd_train_loss[i][best_step:] = [gd_train_loss[i][best_step]] * (
        #             len(gd_train_loss[i]) - best_step)
        #
        #     gd_train_prob_dist[i] = gd_train_prob_dist[i] + [gd_train_prob_dist[i][best_step]] * (
        #             len(prob_loss_trans_list[i]) - len(gd_train_prob_dist[i]))
        #     gd_train_prob_dist[i][best_step:] = [gd_train_prob_dist[i][best_step]] * (
        #             len(gd_train_prob_dist[i]) - best_step)
        #
        #     gd_train_acc[i] = gd_train_acc[i] + [gd_train_acc[i][best_step]] * (
        #                 len(acc_trans_list[i]) - len(gd_train_acc[i]))
        #     gd_train_acc[i][best_step:] = [gd_train_acc[i][best_step]] * (
        #             len(gd_train_acc[i]) - best_step)
        #
        #     gd_train_top_3_freq[i] = gd_train_top_3_freq[i] + [gd_train_top_3_freq[i][best_step]] * (
        #             len(top_3_freq_trans_list[i]) - len(gd_train_top_3_freq[i]))
        #     gd_train_top_3_freq[i][best_step:] = [gd_train_top_3_freq[i][best_step]] * (
        #             len(gd_train_top_3_freq[i]) - best_step)
        #
        #     gd_val_loss[i] = gd_val_loss[i] + [gd_val_loss[i][best_step]] * (
        #                 len(loss_trans_list[i]) - len(gd_val_loss[i]))
        #     gd_val_loss[i][best_step:] = [gd_val_loss[i][best_step]] * (
        #             len(gd_val_loss[i]) - best_step)
        #
        #     gd_val_prob_dist[i] = gd_val_prob_dist[i] + [gd_val_prob_dist[i][best_step]] * (
        #             len(prob_loss_trans_list[i]) - len(gd_val_prob_dist[i]))
        #     gd_val_prob_dist[i][best_step:] = [gd_val_prob_dist[i][best_step]] * (
        #             len(gd_val_prob_dist[i]) - best_step)
        #
        #     gd_val_acc[i] = gd_val_acc[i] + [gd_val_acc[i][best_step]] * (
        #             len(acc_trans_list[i]) - len(gd_val_acc[i]))
        #     gd_val_acc[i][best_step:] = [gd_val_acc[i][best_step]] * (
        #             len(gd_val_acc[i]) - best_step)
        #
        #     gd_val_top_3_freq[i] = gd_val_top_3_freq[i] + [gd_val_top_3_freq[i][best_step]] * (
        #             len(top_3_freq_trans_list[i]) - len(gd_val_top_3_freq[i]))
        #     gd_val_top_3_freq[i][best_step:] = [gd_val_top_3_freq[i][best_step]] * (
        #             len(gd_val_top_3_freq[i]) - best_step)

        # Loss of GD Training
        display_training(save_path, gd_train_loss, trained_tf=loss_trans_list,
                         num_iter_os=len(loss_trans_list[0]),
                         plot_title=None,
                         title="{}/gd_training_loss_es".format(model_kernel) if early_stopping else
                         "{}/gd_training_loss".format(model_kernel), single_seeds_gd=False,
                         single_seeds_tf=False,
                         x_label='Training Epochs',
                         y_label='Negative Log-Likelihood', yscale_log=False, y_lim_l=0,
                         y_lim_u=4,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # Loss of GD on Test Data
        display_training(save_path, losses_gd_list, trained_tf=loss_trans_list,
                         num_iter_os=len(loss_trans_list[0]),
                         plot_title=None,
                         title="{}/trained_tf_loss_es".format(model_kernel) if early_stopping else
                         "{}/trained_tf_loss".format(model_kernel), single_seeds_gd=False,
                         single_seeds_tf=True,
                         x_label='Training Epochs',
                         y_label='Negative Log-Likelihood', yscale_log=False, y_lim_l=0,
                         y_lim_u=4,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # if not imagenet:
        #     # Probability Loss of GD Training
        #     display_training(save_path, gd_train_prob_dist, trained_tf=prob_loss_trans_list,
        #                      num_iter_os=len(prob_loss_trans_list[0]),
        #                      plot_title=None,
        #                      title="{}/gd_training_prob_mse_es".format(model_kernel) if early_stopping else
        #                      "{}/gd_training_prob_mse".format(model_kernel), single_seeds_gd=False,
        #                      single_seeds_tf=False,
        #                      x_label='Training Epochs',
        #                      y_label='MSE on Category Probabilities', yscale_log=False, y_lim_l=0,
        #                      y_lim_u=0.2,
        #                      color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)
        #
        #     # Probability Loss of GD on Test Data
        #     # display_training(save_path, prob_loss_gd_list, trained_tf=prob_loss_trans_list,
        #     #                  num_iter_os=len(prob_loss_trans_list[0]),
        #     #                  plot_title=None,
        #     #                  title="{}/trained_tf_prob_mse_es".format(model_kernel) if early_stopping else
        #     #                  "{}/trained_tf_prob_mse".format(model_kernel), single_seeds_gd=False,
        #     #                  single_seeds_tf=True,
        #     #                  x_label='Training Epochs',
        #     #                  y_label='MSE on Category Probabilities', yscale_log=False, y_lim_l=0,
        #     #                  y_lim_u=0.2,
        #     #                  color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # Accuracy of GD Training
        display_training(save_path, gd_train_acc, trained_tf=acc_trans_list,
                         # max_prob=max_prob,
                         num_iter_os=len(acc_trans_list[0]),
                         plot_title=None,
                         title="{}/gd_training_acc_es".format(model_kernel) if early_stopping else
                         "{}/gd_training_acc".format(model_kernel), single_seeds_gd=False, single_seeds_tf=False,
                         x_label='Training Epochs',
                         y_label='Accuracy', yscale_log=False, y_lim_l=0,
                         y_lim_u=1,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # Accuracy of GD on Test Data
        display_training(save_path, acc_gd_list, trained_tf=acc_trans_list,
                         # max_prob=max_prob,
                         num_iter_os=len(acc_trans_list[0]),
                         plot_title=None,
                         title="{}/trained_tf_acc_es".format(model_kernel) if early_stopping else
                         "{}/trained_tf_acc".format(model_kernel), single_seeds_gd=False, single_seeds_tf=True,
                         x_label='Training Epochs',
                         y_label='Accuracy', yscale_log=False, y_lim_l=0,
                         y_lim_u=1,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # # Top 3 Frequency of GD Training
        # display_training(save_path, gd_train_top_3_freq, trained_tf=top_3_freq_trans_list,
        #                  num_iter_os=len(top_3_freq_gd_list[0]),
        #                  plot_title=None,
        #                  title="{}/gd_training_top_3_freq_es".format(model_kernel) if early_stopping else
        #                  "{}/gd_training_top_3_freq".format(model_kernel), single_seeds_gd=False,
        #                  single_seeds_tf=False,
        #                  x_label='Training Epochs',
        #                  y_label='Top 3 Frequency', yscale_log=False, y_lim_l=0,
        #                  y_lim_u=1,
        #                  color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)
        #
        # # Top 3 Frequency of GD on Test Data
        # display_training(save_path, top_3_freq_gd_list, trained_tf=top_3_freq_trans_list,
        #                  num_iter_os=len(top_3_freq_gd_list[0]),
        #                  plot_title=None,
        #                  title="{}/trained_tf_top_3_freq_es".format(model_kernel) if early_stopping else
        #                  "{}/trained_tf_top_3_freq".format(model_kernel), single_seeds_gd=False,
        #                  single_seeds_tf=True,
        #                  x_label='Training Epochs',
        #                  y_label='Top 3 Frequency', yscale_log=False, y_lim_l=0,
        #                  y_lim_u=1,
        #                  color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)



def plot_linear_training(save_path, data_path, early_stopping=True, linear=False, exp=False, rbf=False,
                  laplacian=False, softmax=False, gd_plus=False, imagenet=False, iterations=5000):
    # kernels = ['linear', 'exp', 'rbf', 'laplacian', 'softmax']
    # kernels = ['laplacian', 'rbf', 'softmax']
    kernels = ['linear', 'exp', 'rbf', 'laplacian', 'softmax']
    labels = ["Linear", "Exp", "RBF", "Laplacian", "Softmax"]

    steps = iterations

    for j, model_kernel in enumerate(kernels):
        if (not linear and model_kernel == 'linear') or (not exp and model_kernel == 'exp') or \
                (not rbf and model_kernel == 'rbf') or (not laplacian and model_kernel == 'laplacian') or \
                (not softmax and model_kernel == 'softmax'):
            continue

        data_f = open(data_path + '/{}/results.json'.format(model_kernel))
        data = json.load(data_f)

        loss_trans_list = np.expand_dims(np.array(data['tf_eval_loss_list']).mean(axis=0), axis=0)
        losses_gd_list = data['tf_gd_eval_loss_list']
        acc_trans_list = np.expand_dims(np.array(data['tf_eval_acc_list']).mean(axis=0), axis=0)
        acc_gd_list = data['tf_gd_eval_acc_list']
        # top_3_freq_trans_list = np.expand_dims(np.array(data['top_3_freq_trans_list']).mean(axis=0), axis=0)
        # top_3_freq_gd_list = data['top_3_freq_gd_list']
        # max_prob = data['max_prob_list']

        # print(max_prob)

        gd_train_loss = data['gd_eval_loss_list']
        # if not imagenet:
        #     gd_train_prob_dist = data['gd_train_prob_dist_list']
        gd_train_acc = data['gd_eval_acc_list']
        # gd_train_top_3_freq = data['gd_train_top_3_freq_list']

        # Loss of GD Training
        display_training(save_path, None, trained_tf=loss_trans_list,
                         num_iter_os=len(loss_trans_list[0]),
                         plot_title=None,
                         title="{}/gd_training_loss_es".format(model_kernel) if early_stopping else
                         "{}/gd_training_loss".format(model_kernel), single_seeds_gd=False,
                         single_seeds_tf=False,
                         x_label='Training Epochs',
                         y_label='Negative Log-Likelihood', yscale_log=False, y_lim_l=0,
                         y_lim_u=4,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # Loss of GD on Test Data
        display_training(save_path, None, trained_tf=loss_trans_list,
                         num_iter_os=len(loss_trans_list[0]),
                         plot_title=None,
                         title="{}/trained_tf_loss_es".format(model_kernel) if early_stopping else
                         "{}/trained_tf_loss".format(model_kernel), single_seeds_gd=False,
                         single_seeds_tf=True,
                         x_label='Training Epochs',
                         y_label='Negative Log-Likelihood', yscale_log=False, y_lim_l=0,
                         y_lim_u=4,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # Accuracy of GD Training
        display_training(save_path, None, trained_tf=acc_trans_list,
                         # max_prob=max_prob,
                         num_iter_os=len(acc_trans_list[0]),
                         plot_title=None,
                         title="{}/gd_training_acc_es".format(model_kernel) if early_stopping else
                         "{}/gd_training_acc".format(model_kernel), single_seeds_gd=False, single_seeds_tf=False,
                         x_label='Training Epochs',
                         y_label='Accuracy', yscale_log=False, y_lim_l=0,
                         y_lim_u=1,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # Accuracy of GD on Test Data
        display_training(save_path, None, trained_tf=acc_trans_list,
                         # max_prob=max_prob,
                         num_iter_os=len(acc_trans_list[0]),
                         plot_title=None,
                         title="{}/trained_tf_acc_es".format(model_kernel) if early_stopping else
                         "{}/trained_tf_acc".format(model_kernel), single_seeds_gd=False, single_seeds_tf=True,
                         x_label='Training Epochs',
                         y_label='Accuracy', yscale_log=False, y_lim_l=0,
                         y_lim_u=1,
                         color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)

        # # Top 3 Frequency of GD Training
        # display_training(save_path, None, trained_tf=top_3_freq_trans_list,
        #                  num_iter_os=len(top_3_freq_trans_list[0]),
        #                  plot_title=None,
        #                  title="{}/gd_training_top_3_freq_es".format(model_kernel) if early_stopping else
        #                  "{}/gd_training_top_3_freq".format(model_kernel), single_seeds_gd=False,
        #                  single_seeds_tf=False,
        #                  x_label='Training Epochs',
        #                  y_label='Top 3 Frequency', yscale_log=False, y_lim_l=0,
        #                  y_lim_u=1,
        #                  color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)
        #
        # # Top 3 Frequency of GD on Test Data
        # display_training(save_path, None, trained_tf=top_3_freq_trans_list,
        #                  num_iter_os=len(top_3_freq_trans_list[0]),
        #                  plot_title=None,
        #                  title="{}/trained_tf_top_3_freq_es".format(model_kernel) if early_stopping else
        #                  "{}/trained_tf_top_3_freq".format(model_kernel), single_seeds_gd=False,
        #                  single_seeds_tf=True,
        #                  x_label='Training Epochs',
        #                  y_label='Top 3 Frequency', yscale_log=False, y_lim_l=0,
        #                  y_lim_u=1,
        #                  color_add=0, loc_first='best', width=6, height=4.5, gd_plus=gd_plus)


def display_training(save_path, gd, trained_tf=None, gd_val=None, max_prob=None,
                     num_iter_os=None, plot_title=None, title=None, single_seeds_gd=False, single_seeds_tf=False,
                     x_label='Training Epochs', y_label='Loss',
                     yscale_log=False, y_lim_l=0, y_lim_u=1,
                     color_add=0, loc_first='upper right', width=4, height=3.5, gd_plus=False, legend2="Trained TF"):
    print("Generating plot: ", title)

    if gd is not None:
        gd_list = gd
        # gd_std = np.std(gd, axis=0)
        # gd = np.mean(gd, axis=0)
        gd_std = np.std(gd, axis=0)
        gd = np.mean(gd, axis=0)

    if trained_tf is not None:
        trained_tf_list = trained_tf
        trained_tf_std = np.std(trained_tf, axis=0)
        trained_tf = np.mean(trained_tf, axis=0)

    if gd_val is not None:
        gd_val_list = gd_val
        gd_val_std = np.std(gd_val, axis=0)
        gd_val = np.mean(gd_val, axis=0)

    if max_prob is not None:
        max_prob_list = max_prob
        max_prob = np.mean(max_prob, axis=0)

    # create figure and axes
    fig, ax1 = pl.subplots()
    ax1.set_xlabel(x_label)
    fig.set_size_inches(width, height)

    # Epochs
    x_range = np.arange(0, num_iter_os)

    # GD
    if gd is not None:
        if single_seeds_gd:
            for s in gd_list:
                ax1.plot(x_range, s, color=colors(0.1 + color_add), alpha=0.2, linewidth='2')
        else:
            ax1.fill_between(x_range, gd - gd_std, gd + gd_std, alpha=0.2,
                             facecolor=colors(0.1 + color_add))
        ax1.plot(x_range, gd, color=colors(0.1 + color_add), label='GD++' if gd_plus else 'GD', linewidth='3')

    # trained tf
    if trained_tf is not None:
        if single_seeds_tf:
            for s in trained_tf_list:
                ax1.plot(x_range, s, color=colors(0.3 + color_add), alpha=0.2, linewidth='2')
        else:
            ax1.fill_between(x_range, trained_tf - trained_tf_std, trained_tf + trained_tf_std, alpha=0.2,
                             facecolor=colors(0.3 + color_add))

        ax1.plot(x_range, trained_tf, color=colors(0.3 + color_add), label=legend2, linewidth='3')

    # gd_val
    if gd_val is not None:
        if single_seeds_gd:
            for s in gd_val_list:
                ax1.plot(x_range, s, color=colors(0.2 + color_add), alpha=0.2, linewidth='2')
        else:
            ax1.fill_between(x_range, gd_val - gd_val_std, gd_val + gd_val_std, alpha=0.2,
                             facecolor=colors(0.2 + color_add))

        ax1.plot(x_range, gd_val, color=colors(0.2 + color_add), label='GD - Validation', linewidth='3')

    if max_prob is not None:
        ax1.plot(x_range, max_prob, color='k', label='Mean Max Probability', linewidth='3')

    ax1.legend(loc='best', framealpha=1, facecolor='white')
    ax1.spines['right'].set_visible(False)

    legend1 = ax1.legend(loc=loc_first, framealpha=0.99, facecolor='white')
    legend1.set_zorder(100)

    plt.ylabel(y_label)
    plt.ylim(y_lim_l, y_lim_u)

    ax1.spines['top'].set_visible(False)

    if plot_title is not None:
        plt.title(plot_title)

    if yscale_log:
        ax1.set_yscale("log")

    plt.tight_layout()

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plt.savefig(save_path + "/" + title + ".jpg", format="jpg", bbox_inches="tight", pad_inches=0)

    plt.close()

    print("Done generating plot")


if __name__ == '__main__':
    if sys.argv[1] == 'plot_training':
        plot_training(sys.argv[2], sys.argv[3], bool(int(sys.argv[4])), bool(int(sys.argv[5])), bool(int(sys.argv[6])),
                      bool(int(sys.argv[7])), bool(int(sys.argv[8])), bool(int(sys.argv[9])), bool(int(sys.argv[10])),
                      bool(int(sys.argv[11])), int(sys.argv[12]))
    if sys.argv[1] == 'plot_linear_training':
        plot_linear_training(sys.argv[2], sys.argv[3], bool(int(sys.argv[4])), bool(int(sys.argv[5])), bool(int(sys.argv[6])),
                      bool(int(sys.argv[7])), bool(int(sys.argv[8])), bool(int(sys.argv[9])), bool(int(sys.argv[10])),
                      bool(int(sys.argv[11])), int(sys.argv[12]))
    elif sys.argv[1] == 'plot_layer_comp':
        compare_layers(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], bool(int(sys.argv[11])))
    elif sys.argv[1] == 'context_comp':
        context_comparison(sys.argv[2], bool(int(sys.argv[3])), bool(int(sys.argv[4])), bool(int(sys.argv[5])),
                           bool(int(sys.argv[6])), bool(int(sys.argv[7])), bool(int(sys.argv[8])), int(sys.argv[9]))
    elif sys.argv[1] == 'compare_kernels':
        compare_kernels(sys.argv[2], bool(int(sys.argv[3])), bool(int(sys.argv[4])), bool(int(sys.argv[5])),
                           bool(int(sys.argv[6])), bool(int(sys.argv[7])), bool(int(sys.argv[8])), bool(int(sys.argv[9])))
