import os
import sys
import numpy as np
import json


def write_metrics_to_txt():
    layers = [1, 2, 3, 4, 5]
    kernels = ['softmax']
    datapaths = ['results/reg_emb_cats_25_d_10/layers_2/l_2048',
                 'results/reg_emb_cats_25_d_10/layers_2/l_4096',
                 'results/reg_emb_cats_25_d_10/layers_2/l_8192',
                 'results/reg_emb_cats_25_d_10/layers_2/l_12288',
                 'results/reg_emb_cats_25_d_10/layers_2/l_16384']
    training_data = [2048, 4096, 8192, 12288, 16384]
    savepath = "results/reg_emb_cats_25_d_10/metrics_training_data.txt"
    steps = 50

    imagenet = False
    early_stopping = True

    file = open(savepath, "w")
    for j, model_kernel in enumerate(kernels):
        file.write("Kernel: {}\n".format(model_kernel))
        for layer, data_path in enumerate(datapaths):
            data_f = open(data_path + '/{}/results.json'.format(model_kernel))
            data = json.load(data_f)

            loss_trans_list = list(filter(None, data['loss_trans_list']))
            losses_gd_list = list(filter(None, data['losses_gd_list']))
            if not imagenet:
                prob_loss_trans_list = list(filter(None, data['prob_dist_trans_list']))
                prob_loss_gd_list = list(filter(None, data['prob_dist_gd_list']))
            acc_trans_list = list(filter(None, data['acc_trans_list']))
            acc_gd_list = list(filter(None, data['acc_gd_list']))
            top_3_freq_trans_list = list(filter(None, data['top_3_freq_trans_list']))
            top_3_freq_gd_list = list(filter(None, data['top_3_freq_gd_list']))

            gd_train_loss = list(filter(None, data['gd_train_loss_list']))
            if not imagenet:
                gd_train_prob_dist = list(filter(None, data['gd_train_prob_dist_list']))
            gd_train_acc = list(filter(None, data['gd_train_acc_list']))
            gd_train_top_3_freq = list(filter(None, data['gd_train_top_3_freq_list']))

            if early_stopping:
                best_idx_trans_list = data['best_idx_trans_list']
                for i, idx in enumerate(best_idx_trans_list):
                    idx = idx // 100

                    loss_trans_list[i][idx + 1:] = [loss_trans_list[i][idx]] * (steps - idx - 1)

                    if not imagenet:
                        prob_loss_trans_list[i][idx + 1:] = [prob_loss_trans_list[i][idx]] * (
                                steps - idx - 1)

                    acc_trans_list[i][idx + 1:] = [acc_trans_list[i][idx]] * (steps - idx - 1)

                    top_3_freq_trans_list[i][idx + 1:] = [top_3_freq_trans_list[i][idx]] * (steps - idx - 1)

                best_idx_gd_list = data['gd_val_best_step']
                for i, idx in enumerate(best_idx_gd_list):
                    idx = idx // 100

                    gd_train_loss[i][idx + 1:] = [gd_train_loss[i][idx]] * (steps - idx - 1)

                    if not imagenet:
                        gd_train_prob_dist[i][idx + 1:] = [gd_train_prob_dist[i][idx]] * (
                                steps - idx - 1)

                    gd_train_acc[i][idx + 1:] = [gd_train_acc[i][idx]] * (steps - idx - 1)

                    gd_train_top_3_freq[i][idx + 1:] = [gd_train_top_3_freq[i][idx]] * (steps - idx - 1)

            file.write("Training Data: {}\n".format(training_data[layer]))
            file.write("Trained TF Loss: {}\n".format(np.mean(loss_trans_list, axis=0)[-1]))
            file.write("Trained TF Accuracy: {}\n".format(np.mean(acc_trans_list, axis=0)[-1]))
            file.write("Trained TF Top 3 Frequency: {}\n".format(np.mean(top_3_freq_trans_list, axis=0)[-1]))
            file.write(
                "Trained TF MSE on Category Probabilities: {}\n".format(np.mean(prob_loss_trans_list, axis=0)[-1]))
            file.write("---------------------------\n")
            file.write("GD Loss: {}\n".format(np.mean(gd_train_loss, axis=0)[-1]))
            file.write("GD Accuracy: {}\n".format(np.mean(gd_train_acc, axis=0)[-1]))
            file.write("GD Top 3 Frequency: {}\n".format(np.mean(gd_train_top_3_freq, axis=0)[-1]))
            file.write("GD MSE on Category Probabilities: {}\n".format(np.mean(gd_train_prob_dist, axis=0)[-1]))
            file.write("---------------------------\n")
    file.close()


if __name__ == '__main__':
    write_metrics_to_txt()
