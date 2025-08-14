"""
Data and weight generation.
Functions to create synthetic classification datasets.
"""
import math
import numpy as np

import os
import sys

from config import config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


class HighDimCategorical(Dataset):
    def __init__(self, device, model_type, seed, batches, i_size, c_size, data_e_size, params_e_size, cats,
                 k, dist, l, W_e, input_range):
        """Create a high-dimensional classification dataset using a grid, where x ~ U(-1, 1)"""
        self.batches = batches
        self.data = torch.empty((batches, i_size + 2 * cats + params_e_size, c_size + 1))
        self.labels = torch.empty((batches, c_size + 1))

        for b in range(batches):
            generator = torch.Generator().manual_seed(seed + b)

            if W_e is None:
                W_e = torch.randn((data_e_size, cats), generator=generator)

            # draw training data and query
            x = torch.empty((i_size, c_size)).uniform_(-1 * input_range, input_range, generator=generator)
            x_query = torch.empty((i_size, 1)).uniform_(-1 * input_range, input_range, generator=generator)

            c_idx = torch.randperm(cats, generator=generator)[:k]

            w_c = W_e[:, c_idx]

            x_mean = torch.randn((i_size, k), generator=generator)

            l2_means = torch.sum((x_mean.unsqueeze(dim=2) - x_mean.unsqueeze(dim=1)) ** 2, dim=0)
            modified_l2 = torch.where(l2_means == 0, torch.inf, l2_means)

            min_indices = torch.argmin(modified_l2, dim=1)
            closest = modified_l2.min(dim=1).values.unsqueeze(dim=-1)

            sigma = torch.sqrt(-1 * closest / torch.log(torch.tensor(dist)))

            diff = x_mean.unsqueeze(dim=-1) - x.unsqueeze(dim=1)
            l2 = torch.sum(diff ** 2, dim=0)
            rbf = torch.exp(-1 * l2 / (sigma ** 2))

            f = l * torch.matmul(w_c, rbf)
            logits = torch.matmul(f.T, W_e)
            probs = F.softmax(logits, dim=1)

            y_data = torch.multinomial(probs, num_samples=1).squeeze(dim=1)
            v_data_full = F.one_hot(y_data, num_classes=cats).permute(1, 0)

            diff_target = x_mean.unsqueeze(dim=-1) - x_query.unsqueeze(dim=1)
            l2_target = torch.sum(diff_target ** 2, dim=0)
            rbf_target = torch.exp(-1 * l2_target / (sigma ** 2))

            f_target = l * torch.matmul(w_c, rbf_target)
            logits_target = torch.matmul(f_target.T, W_e)
            probs_target = F.softmax(logits_target, dim=1)

            y_target = torch.multinomial(probs_target, num_samples=1, generator=generator).squeeze(dim=1)

            v_target_full = F.one_hot(y_target, num_classes=cats).permute(1, 0)

            if model_type in ('interleaved', 'feedforward', 'moe'):
                seq = torch.cat(
                    [x, v_data_full, torch.zeros(cats, c_size) + 1 / cats, torch.zeros(params_e_size, c_size)],
                    dim=0)
                target = torch.cat(
                    [x_query, v_target_full, torch.zeros(cats, 1) + 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                zero = torch.cat(
                    [x_query, torch.zeros(cats, 1), torch.zeros(cats, 1) + 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                seq = torch.cat([seq, zero], dim=1)
            elif model_type == 'linear_approx':
                seq = torch.cat(
                    [x, v_data_full - 1 / cats, torch.zeros(params_e_size, c_size)],
                    dim=0)
                target = torch.cat(
                    [x_query, v_target_full - 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                zero = torch.cat(
                    [x_query, torch.zeros(cats, 1) - 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                seq = torch.cat([seq, zero], dim=1)

            self.data[b] = seq
            self.labels[b] = torch.concatenate([y_data, y_target], dim=0)

        self.data = self.data.to(device)
        self.labels = self.labels.type(torch.LongTensor).to(device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.batches


class HighDimMixtureCategorical(Dataset):
    def __init__(self, device, model_type, seed, batches, i_size, c_size, data_e_size, params_e_size, cats,
                 k, dist, l, W_e, J, u_var, input_range):
        """Create a high-dimensional classification dataset using a grid, where x ~ U(-1, 1)"""
        self.batches = batches
        self.data = torch.empty((batches, i_size + 2 * cats + params_e_size, c_size + 1))
        self.labels = torch.empty((batches, c_size + 1))

        generator = torch.Generator().manual_seed(seed + batches)
        u_matrix = torch.normal(mean=torch.zeros((J, cats)), std=u_var*torch.ones((J, cats)), generator=generator)

        for b in range(batches):
            generator = torch.Generator().manual_seed(seed + b)

            if W_e is None:
                W_e = torch.randn((data_e_size, cats), generator=generator)

            # draw training data and query
            x = torch.empty((i_size, c_size)).uniform_(-1 * input_range, input_range, generator=generator)
            x_query = torch.empty((i_size, 1)).uniform_(-1 * input_range, input_range, generator=generator)

            c_idx = self.get_c_idx(J=J, u_matrix=u_matrix, k=k, generator=generator)

            w_c = W_e[:, c_idx]

            x_mean = torch.randn((i_size, k), generator=generator)

            l2_means = torch.sum((x_mean.unsqueeze(dim=2) - x_mean.unsqueeze(dim=1)) ** 2, dim=0)
            modified_l2 = torch.where(l2_means == 0, torch.inf, l2_means)

            min_indices = torch.argmin(modified_l2, dim=1)
            closest = modified_l2.min(dim=1).values.unsqueeze(dim=-1)

            sigma = torch.sqrt(-1 * closest / torch.log(torch.tensor(dist)))

            diff = x_mean.unsqueeze(dim=-1) - x.unsqueeze(dim=1)
            l2 = torch.sum(diff ** 2, dim=0)
            rbf = torch.exp(-1 * l2 / (sigma ** 2))

            f = l * torch.matmul(w_c, rbf)
            logits = torch.matmul(f.T, W_e)
            probs = F.softmax(logits, dim=1)

            y_data = torch.multinomial(probs, num_samples=1).squeeze(dim=1)
            v_data_full = F.one_hot(y_data, num_classes=cats).permute(1, 0)

            diff_target = x_mean.unsqueeze(dim=-1) - x_query.unsqueeze(dim=1)
            l2_target = torch.sum(diff_target ** 2, dim=0)
            rbf_target = torch.exp(-1 * l2_target / (sigma ** 2))

            f_target = l * torch.matmul(w_c, rbf_target)
            logits_target = torch.matmul(f_target.T, W_e)
            probs_target = F.softmax(logits_target, dim=1)

            y_target = torch.multinomial(probs_target, num_samples=1, generator=generator).squeeze(dim=1)

            v_target_full = F.one_hot(y_target, num_classes=cats).permute(1, 0)

            if model_type in ('interleaved', 'feedforward', 'moe'):
                seq = torch.cat(
                    [x, v_data_full, torch.zeros(cats, c_size) + 1 / cats, torch.zeros(params_e_size, c_size)],
                    dim=0)
                target = torch.cat(
                    [x_query, v_target_full, torch.zeros(cats, 1) + 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                zero = torch.cat(
                    [x_query, torch.zeros(cats, 1), torch.zeros(cats, 1) + 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                seq = torch.cat([seq, zero], dim=1)
            elif model_type == 'linear_approx':
                seq = torch.cat(
                    [x, v_data_full - 1 / cats, torch.zeros(params_e_size, c_size)],
                    dim=0)
                target = torch.cat(
                    [x_query, v_target_full - 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                zero = torch.cat(
                    [x_query, torch.zeros(cats, 1) - 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                seq = torch.cat([seq, zero], dim=1)

            self.data[b] = seq
            self.labels[b] = torch.concatenate([y_data, y_target], dim=0)

        self.data = self.data.to(device)
        self.labels = self.labels.type(torch.LongTensor).to(device)

    def get_c_idx(self, J, u_matrix, k, generator):
        pi = torch.softmax(u_matrix, dim=1)

        mu = torch.randperm(J, generator=generator)[0]
        # print(mu)
        # print(pi[mu])

        c_idx = torch.multinomial(pi[mu], num_samples=k, replacement=False, generator=generator)
        # print(c_idx)

        return c_idx

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.batches


class RandomGridData(Dataset):
    def __init__(self, device, model_type, seed, batches, i_size, c_size, data_e_size, params_e_size, cats, W_e,
                 input_range):
        """Create a classification dataset using a grid, where x ~ U(-1, 1)"""
        self.batches = batches
        self.data = torch.empty((batches, i_size + 2 * cats + params_e_size, c_size + 1))
        self.labels = torch.empty((batches, c_size + 1))

        for b in range(batches):
            generator = torch.Generator().manual_seed(seed + b)

            if W_e is None:
                W_e = torch.randn((data_e_size, cats), generator=generator)

            # Random input x and query
            x = (torch.rand(i_size, c_size, generator=generator) * (2 * input_range) - input_range)
            x_query = (torch.rand(i_size, 1, generator=generator) * (2 * input_range) - input_range)

            # Randomly choose indices without replacement
            c_idx = torch.randperm(cats, generator=generator)[:4]
            c = W_e[:, c_idx]  # shape: (embedding_dim, 4)

            # # Quadrant computation
            x_pos = (x[0, :] >= 0).long()
            y_pos = (x[1, :] >= 0).long()

            quad = torch.where(y_pos == 1, 1 - x_pos, 2 + x_pos)

            # Calculate f
            f = c[:, quad]  # shape: (embedding_dim, c_size)

            # Probability logits
            logits = torch.matmul(f.T, W_e)
            probs = F.softmax(logits, dim=1)  # shape: (c_size, cats)

            # Draw labels for each sample
            y_data = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(dim=1)

            # One-hot labels
            v_data_full = F.one_hot(y_data, num_classes=cats).float().permute(1, 0)

            # Query quadrant
            x_pos_query = (x_query[0, :] >= 0).long()
            y_pos_query = (x_query[1, :] >= 0).long()

            quad_query = torch.where(y_pos_query == 1, 1 - x_pos_query, 2 + x_pos_query)

            # f(x_query)
            f_target = c[:, quad_query]

            # Query probs
            logits_target = torch.matmul(f_target.T, W_e)
            probs_target = F.softmax(logits_target, dim=1)

            # Draw query label
            # y_target = torch.distributions.Categorical(logits=f_target.T @ W_e, generator=generator).sample()
            y_target = torch.multinomial(probs_target, num_samples=1, generator=generator).squeeze(dim=1)

            # One-hot query labels
            v_target_full = F.one_hot(y_target, num_classes=cats).float().permute(1, 0)

            # Sequences and embeddings
            W_e_seq = W_e[:, y_data].T
            w_e_target = W_e[:, y_target].T

            if model_type in ('interleaved', 'feedforward', 'moe'):
                seq = torch.cat([x, v_data_full, torch.zeros(cats, c_size) + 1 / cats, torch.zeros(params_e_size, c_size)],
                                dim=0)
                target = torch.cat([x_query, v_target_full, torch.zeros(cats, 1) + 1 / cats, torch.zeros(params_e_size, 1)],
                                   dim=0)
                zero = torch.cat(
                    [x_query, torch.zeros(cats, 1), torch.zeros(cats, 1) + 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                seq = torch.cat([seq, zero], dim=1)

            self.data[b] = seq
            self.labels[b] = torch.concatenate([y_data, y_target], dim=0)

        self.data = self.data.to(device)
        self.labels = self.labels.type(torch.LongTensor).to(device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.batches


class ImageNet(Dataset):
    def __init__(self, device, model_type, seed, batches, data, params_e_size, cats=5, examples_per_cat=10):
        """Create a high-dimensional classification dataset using a grid, where x ~ U(-1, 1)"""
        c_size = cats * examples_per_cat
        i_size = data.shape[2]  # dimensionality of each feature
        num_cats = data.shape[0]  # number of imagenet classes

        self.batches = batches
        self.data = torch.empty((batches, i_size + 2 * cats + params_e_size, c_size + 1))
        self.labels = torch.empty((batches, c_size + 1))

        for b in range(batches):
            generator = torch.Generator().manual_seed(seed + b)

            x = torch.zeros((i_size, c_size + 1))
            y = torch.zeros((1, c_size + 1))

            classes = torch.randperm(num_cats, generator=generator)[:cats]
            query_class = classes[-1]
            query_idx = cats - 1

            # non-query classes
            for i in range(cats - 1):
                # randomly draw examples_per_cat examples of each class
                cur_label = classes[i]
                idx = torch.randperm(data[cur_label].shape[0], generator=generator)[:examples_per_cat]
                x[:, i * examples_per_cat:(i + 1) * examples_per_cat] = data[cur_label, idx].T
                y[:, i * examples_per_cat:(i + 1) * examples_per_cat] = torch.ones(examples_per_cat) * i

            # query classes
            # randomly draw examples_per_cat examples of query class for context, and one additional example for query
            idx = torch.randperm(data[query_class].shape[0], generator=generator)[:examples_per_cat + 1]
            x[:, -(examples_per_cat + 1):] = data[query_class, idx].T
            y[:, -(examples_per_cat + 1):] = torch.ones(examples_per_cat + 1) * query_idx

            features = torch.cat([x, y], dim=0)
            features_context = features[:, :-1]
            features_query = features[:, -1:]

            # permute context
            shuffled_indices = torch.randperm(c_size, generator=generator)
            features_context = features_context[:, shuffled_indices]

            # get labels
            y_data = features_context[-1, :].long()
            y_target = features_query[-1].long()

            # get final features
            features_context = features_context[:-1, :]
            features_query = features_query[:-1, :]

            # get one-hot encodings of labels
            v_data_full = F.one_hot(y_data, num_classes=cats).permute(1, 0)
            v_target_full = F.one_hot(y_target, num_classes=cats).permute(1, 0)

            if model_type in ('interleaved', 'feedforward'):
                seq = torch.cat([features_context, v_data_full, torch.zeros(cats, c_size) + 1 / cats,
                                 torch.zeros(params_e_size, c_size)],
                                dim=0)
                target = torch.cat(
                    [features_query, v_target_full, torch.zeros(cats, 1) + 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                zero = torch.cat(
                    [features_query, torch.zeros(cats, 1), torch.zeros(cats, 1) + 1 / cats,
                     torch.zeros(params_e_size, 1)], dim=0)
                seq = torch.cat([seq, zero], dim=1)
            elif model_type == 'linear_approx':
                seq = torch.cat(
                    [features_context, v_data_full - 1 / cats, torch.zeros(params_e_size, c_size)],
                    dim=0)
                target = torch.cat(
                    [features_query, v_target_full - 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                zero = torch.cat(
                    [features_query, torch.zeros(cats, 1) - 1 / cats, torch.zeros(params_e_size, 1)],
                    dim=0)
                seq = torch.cat([seq, zero], dim=1)

            self.data[b] = seq
            self.labels[b] = torch.concatenate([y_data, y_target], dim=0)

        self.data = self.data.to(device)
        self.labels = self.labels.type(torch.LongTensor).to(device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.batches


def get_train_features(data_path):
    model = VGG16(weights='imagenet', include_top=False)
    input_shape = [224, 224, 3]
    target_size = [224, 224]

    labels = os.listdir(data_path)
    folder_names = []
    for i in labels:
        folder_names.append(os.listdir("{}/{}".format(data_path, i)))

    num_folders = len(folder_names)
    max_len = 0
    for i in range(len(folder_names)):
        max_len = max(max_len, len(folder_names[i]))

    print("maximum number of images: ", max_len)

    x = np.zeros((num_folders, max_len, 512))
    print("extracting training features...")
    for i in range(len(folder_names)):
        x_temp = np.zeros((len(folder_names[i]), input_shape[0], input_shape[1], input_shape[2]))
        idx = 0
        for img_path in folder_names[i]:
            img = keras.utils.load_img("{}/{}/{}".format(data_path, labels[i], img_path),
                                       target_size=(target_size[0], target_size[1]))
            img_data = keras.utils.img_to_array(img)

            x_temp[idx] = img_data

            idx += 1

        x_temp = preprocess_input(x_temp)

        features = model.predict(x_temp)
        features_mean = np.mean(features, axis=(1, 2))

        x = x.at[i, :len(folder_names[i]), :].set(features_mean)
        print("{} out of {}".format(i + 1, num_folders))

    print("done extracting training features with shape ", x.shape)

    print("saving training features to train_features.npy ...")
    np.save("train_features", x)
    print("done saving train features")

    return x


def get_val_features(data_path):
    model = VGG16(weights='imagenet', include_top=False)
    input_shape = [224, 224, 3]
    target_size = [224, 224]

    labels = os.listdir(data_path)
    folder_names = []
    for i in labels:
        folder_names.append(os.listdir("{}/{}".format(data_path, i)))

    num_folders = len(folder_names)
    max_len = 0
    for i in range(len(folder_names)):
        max_len = max(max_len, len(folder_names[i]))

    print("maximum number of images: ", max_len)

    x = np.zeros((num_folders, max_len, 512))
    print("extracting val features...")
    for i in range(len(folder_names)):
        x_temp = np.zeros((len(folder_names[i]), input_shape[0], input_shape[1], input_shape[2]))
        idx = 0
        for img_path in folder_names[i]:
            img = keras.utils.load_img("{}/{}/{}".format(data_path, labels[i], img_path),
                                       target_size=(target_size[0], target_size[1]))
            img_data = keras.utils.img_to_array(img)

            x_temp[idx] = img_data

            idx += 1

        x_temp = preprocess_input(x_temp)

        features = model.predict(x_temp)
        features_mean = np.mean(features, axis=(1, 2))

        x = x.at[i, :len(folder_names[i]), :].set(features_mean)
        print("{} out of {}".format(i + 1, num_folders))

    print("done extracting val features with shape ", x.shape)

    print("saving val features to val_features.npy ...")
    np.save("val_features", x)
    print("done saving val features")

    return x

