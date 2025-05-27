"""
Data and weight generation.
Functions to create synthetic classification datasets.
"""
import math
from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np

import os
import sys

from config import config

import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils

P = PositionalSharding


def create_cat_data_grid(rng, i_size, c_size, e_size, cats, input_range, w_scale, bias=False):
    """Create a classification dataset using a grid, where x ~ U(-1, 1)"""
    cat_0_logits = jnp.array([1 + jnp.log(8), 1, 1])
    cat_1_logits = jnp.array([1, 1 + jnp.log(8), 1])
    cat_2_logits = jnp.array([1, 1, 1 + jnp.log(8)])

    cat_0 = cat_0_logits
    cat_1 = cat_1_logits
    cat_2 = cat_2_logits

    template_1 = jnp.array([cat_0, cat_1, cat_2, cat_0])
    template_2 = jnp.array([cat_0, cat_2, cat_1, cat_0])
    template_3 = jnp.array([cat_1, cat_0, cat_0, cat_2])
    template_4 = jnp.array([cat_2, cat_0, cat_0, cat_1])

    template_5 = jnp.array([cat_1, cat_0, cat_2, cat_1])
    template_6 = jnp.array([cat_1, cat_2, cat_0, cat_1])
    template_7 = jnp.array([cat_2, cat_1, cat_1, cat_0])
    template_8 = jnp.array([cat_0, cat_1, cat_1, cat_2])

    template_9 = jnp.array([cat_2, cat_0, cat_1, cat_2])
    template_10 = jnp.array([cat_2, cat_1, cat_0, cat_2])
    template_11 = jnp.array([cat_0, cat_2, cat_2, cat_1])
    template_12 = jnp.array([cat_1, cat_2, cat_2, cat_0])

    templates = jnp.array(
        [template_1, template_2, template_3, template_4, template_5, template_6, template_7, template_8, template_9,
         template_10, template_11, template_12])

    rng, new_rng, new_rng2, new_rng3, new_rng4, new_rng5 = jax.random.split(rng, num=6)

    # draw x's
    x = jax.random.uniform(rng, shape=[c_size, i_size], minval=-input_range, maxval=input_range)

    template_idx = jax.random.randint(new_rng, [], 0, 12)
    template = templates[template_idx]

    x_0 = x[:, 0]
    x_1 = x[:, 1]

    temp_0 = jnp.where(x_0 >= 0, 2, 0)
    temp_1 = jnp.where(x_1 >= 0, 3, 2)
    temp = jnp.concatenate([temp_0[:, None], temp_1[:, None]], axis=1)
    quad = jnp.sum(temp, axis=1) - 2

    logits = template[quad]

    prob = jax.nn.softmax(logits, axis=1)

    y_data = jax.random.categorical(new_rng3, logits, axis=1)

    # get one-hot encoding
    v_data_full = jax.nn.one_hot(y_data, cats)
    # only keep first cats-1 rows
    v_data = v_data_full[:, :-1]

    x_query = jnp.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
    x_query = jax.random.permutation(new_rng4, x_query, axis=0)

    x_0_query = x_query[:, 0]
    x_1_query = x_query[:, 1]
    temp_0_query = jnp.where(x_0_query >= 0, 2, 0)
    temp_1_query = jnp.where(x_1_query >= 0, 3, 2)
    temp_query = jnp.concatenate([temp_0_query[:, None], temp_1_query[:, None]], axis=1)
    quad_query = jnp.sum(temp_query, axis=1) - 2

    logits_target = template[quad_query]
    prob_target = jax.nn.softmax(logits_target, axis=1)

    y_target = jax.random.categorical(new_rng5, logits_target, axis=1)

    # one-hot encoding of target
    v_target_full = jax.nn.one_hot(y_target, cats)
    v_target = v_target_full[:, :-1]

    E_w_e_init = jnp.zeros(shape=(c_size, e_size))
    f_init = jnp.zeros(shape=(c_size, e_size))

    # format data for output
    seq = jnp.concatenate([x, v_data_full, f_init], -1)
    seq = jnp.tile(seq, (4, 1, 1))  # in-context data sequence

    f_target = logits_target[:, :-1]
    target = jnp.concatenate([x_query, v_target_full, f_target], axis=-1)
    zero = jnp.concatenate([x_query, jnp.zeros((4, cats)), jnp.zeros((4, e_size))],
                           axis=-1)
    seq = jnp.concatenate([seq, zero[:, None, :]], axis=1)

    W_e = jnp.array([[1, 0, 0], [0, 1, 0]]).astype('float32')

    return jnp.squeeze(seq), jnp.squeeze(target), jnp.squeeze(jnp.tile(prob, (4, 1, 1))), jnp.squeeze(prob_target), \
        jnp.squeeze(v_data_full), jnp.squeeze(v_target_full), \
        jnp.squeeze(jnp.tile(W_e, (4, 1, 1))), jnp.squeeze(v_target_full)


data_creator = vmap(create_cat_data_grid,
                    in_axes=(0, None, None, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(1)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=5), 2, 20, 2, 3, 1, 1, True)


def create_w_e(rng, e_size, cats):
    rng, new_rng = jax.random.split(rng, num=2)
    W_e = jax.random.normal(new_rng, shape=(e_size, cats))

    return W_e


def create_cat_data_random_grid(rng, i_size, c_size, e_size, cats, input_range, w_scale, W_e, bias=False):
    """Create a classification dataset using a grid, where x ~ U(-1, 1)"""
    rng, new_rng, new_rng2, new_rng3, new_rng4, new_rng5 = jax.random.split(rng, num=6)

    # draw training data and query
    x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                           minval=-input_range, maxval=input_range)
    x_query = jax.random.uniform(new_rng2, shape=[1, i_size],
                                 minval=-input_range, maxval=input_range)

    c_idx = jax.random.choice(new_rng3, a=cats, shape=(4,), replace=False)
    c = W_e[:, c_idx]

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
    y_data = jax.random.categorical(new_rng4, f.T @ W_e, axis=1)

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
    y_target = jax.random.categorical(new_rng5, f_target.T @ W_e, axis=1)

    v_target_full = jax.nn.one_hot(y_target, num_classes=cats)

    W_e_seq = W_e[:, y_data].T
    E_w_e_init = jnp.zeros(shape=(c_size, e_size))
    f_init = jnp.zeros(shape=(c_size, e_size))

    w_e_target = W_e[:, y_target].T

    seq = jnp.concatenate([x, v_data_full, jnp.zeros(shape=(c_size, cats)) + 1 / cats, f_init], axis=-1)
    target = jnp.concatenate([x_query, v_target_full, jnp.zeros(shape=(1, cats)) + 1 / cats, f_target.T], axis=-1)
    zero = jnp.concatenate([x_query, jnp.zeros((1, cats)), jnp.zeros(shape=(1, cats)) + 1 / cats, jnp.zeros((1, e_size))], axis=-1)
    seq = jnp.concatenate([seq, zero], axis=0)

    return jnp.squeeze(seq), jnp.squeeze(target), jnp.squeeze(probs), jnp.squeeze(probs_target), jnp.squeeze(f), \
        jnp.squeeze(f_target), jnp.squeeze(v_data_full), jnp.squeeze(v_target_full), \
        jnp.squeeze(W_e), jnp.squeeze(v_target_full)


data_creator = vmap(create_cat_data_random_grid,
                    in_axes=(0, None, None, None, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(1)
rng, test_rng_avg = jax.random.split(rng, 2)
W_e = create_w_e(rng, 5, 20)
test_data = data_creator(jax.random.split(rng, num=2), 2, 100, 5, 20, 1, 1.0, W_e, False)


def create_cat_data_high_dim(rng, i_size, c_size, params_e_size, cats, k, dist, l, W_e, input_range):
    """Create a high-dimensional classification dataset using a grid, where x ~ U(-1, 1)"""
    rng, new_rng, new_rng2, new_rng3, new_rng4, new_rng5, new_rng6 = jax.random.split(rng, num=7)

    if W_e is None:
        rng, new_rng7 = jax.random.split(rng, num=2)
        W_e = jax.random.normal(new_rng7, shape=(config.data_e_size, cats))

    # draw training data and query
    x = jax.random.uniform(new_rng, shape=[c_size, i_size], minval=-input_range, maxval=input_range)
    x_query = jax.random.uniform(new_rng2, shape=[1, i_size],
                                 minval=-input_range, maxval=input_range)

    c_idx = jax.random.choice(new_rng3, a=cats, shape=(k,), replace=False)
    w_c = W_e[:, c_idx]

    x_mean = jax.random.normal(new_rng4, shape=(k, i_size))

    l2_means = jnp.sum((x_mean[:, None, :] - x_mean[None, :, :]) ** 2, axis=-1)

    modified_l2 = jnp.where(l2_means == 0, jnp.inf, l2_means)

    # Find the indices of the minimum values along the rows
    min_indices = jnp.argmin(modified_l2, axis=1)

    closest = l2_means[jnp.arange(l2_means.shape[0]), min_indices]

    sigma = jnp.sqrt(-1 * closest / jnp.log(dist))

    diff = x[:, None, :] - x_mean[None, :, :]

    l2 = jnp.sum(diff ** 2, axis=-1)

    rbf = jnp.exp(-1 * l2 / (sigma ** 2))

    f = l * (rbf @ w_c.T)

    logits = f @ W_e

    probs = jax.nn.softmax(logits, axis=1)

    # randomly draw labels for each sample
    y_data = jax.random.categorical(new_rng5, logits, axis=1)

    v_data_full = jax.nn.one_hot(y_data, num_classes=cats)

    diff_target = x_query[:, None, :] - x_mean[None, :, :]

    l2_target = jnp.sum(diff_target ** 2, axis=-1)

    rbf_target = jnp.exp(-1 * l2_target / (sigma ** 2))

    f_target = l * (rbf_target @ w_c.T)

    logits_target = f_target @ W_e

    probs_target = jax.nn.softmax(logits_target, axis=1)

    # randomly draw labels for each sample
    y_target = jax.random.categorical(new_rng6, logits_target, axis=1)

    v_target_full = jax.nn.one_hot(y_target, num_classes=cats)

    seq = jnp.concatenate([x, v_data_full, jnp.zeros((x.shape[0], cats)) + 1 / cats, jnp.zeros((x.shape[0], params_e_size))], axis=-1)
    target = jnp.concatenate([x_query, v_target_full, 1 / cats + jnp.zeros((x_query.shape[0], cats)),
                              jnp.zeros((x_query.shape[0], params_e_size))], axis=-1)
    zero = jnp.concatenate([x_query, jnp.zeros((1, cats)), 1 / cats + jnp.zeros((x_query.shape[0], cats)),
                            jnp.zeros((x_query.shape[0], params_e_size))], axis=-1)
    seq = jnp.concatenate([seq, zero], axis=0)

    return jnp.squeeze(seq), jnp.squeeze(target), jnp.squeeze(probs), jnp.squeeze(probs_target), jnp.squeeze(f), \
        jnp.squeeze(f_target), jnp.squeeze(v_data_full), jnp.squeeze(v_target_full), \
        jnp.squeeze(W_e), jnp.squeeze(v_target_full)


data_creator = vmap(create_cat_data_high_dim,
                    in_axes=(0, None, None, None, None, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(1)
rng, test_rng_avg = jax.random.split(rng, 2)
W_e = create_w_e(rng, 25, 100)

# sharding = PositionalSharding(mesh_utils.create_device_mesh((4, 1)))
# test_data = data_creator(jax.device_put(jax.random.split(rng, num=400), sharding.reshape(4, 1)),
#                          10, 500, 25, 100, 5, 0.1, 10, W_e, 1)
# test_data = list(test_data)
# print(test_data[0].shape)
# test_data[0] = jax.device_put(test_data[0], sharding.reshape(4, 1, 1))
# test_data = tuple(test_data)


# preprocess all images (extract all features beforehand)
# training data: make giant array with all features
# 2-D: (num classes, num images per class, 512)
# also save dict with keys = wnid and values = integer idx
# for val data, use the dict to process features
# in data: randomly select 5 classes

def get_train_features():
    data_path = config.data_path
    model = config.model
    input_shape = config.input_shape
    target_size = config.target_size

    labels = os.listdir(data_path)
    folder_names = []
    for i in labels:
        folder_names.append(os.listdir("{}/{}".format(data_path, i)))

    num_folders = len(folder_names)
    max_len = 0
    for i in range(len(folder_names)):
        max_len = max(max_len, len(folder_names[i]))

    print("maximum number of images: ", max_len)

    x = jnp.zeros((num_folders, max_len, 512))
    train_lens = []
    print("extracting training features...")
    for i in range(len(folder_names)):
        x_temp = np.zeros((len(folder_names[i]), input_shape[0], input_shape[1], input_shape[2]))
        idx = 0
        train_lens.append(len(folder_names[i]))
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
    jnp.save("train_features", x)

    train_lens = jnp.array(train_lens)
    jnp.save("train_lens", train_lens)
    print("done saving train features")

    return x


def get_val_features():
    data_path = config.data_path_test
    model = config.model
    input_shape = config.input_shape
    target_size = config.target_size

    labels = os.listdir(data_path)
    folder_names = []
    for i in labels:
        folder_names.append(os.listdir("{}/{}".format(data_path, i)))

    num_folders = len(folder_names)
    max_len = 0
    for i in range(len(folder_names)):
        max_len = max(max_len, len(folder_names[i]))

    print("maximum number of images: ", max_len)

    x = jnp.zeros((num_folders, max_len, 512))
    val_lens = []
    print("extracting val features...")
    for i in range(len(folder_names)):
        x_temp = np.zeros((len(folder_names[i]), input_shape[0], input_shape[1], input_shape[2]))
        idx = 0
        val_lens.append(len(folder_names[i]))
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
    jnp.save("val_features", x)

    val_lens = jnp.array(val_lens)
    jnp.save("val_lens", val_lens)
    print("done saving val features")

    return x


def create_img_data_np(seed, data, data_len, cats=5, examples_per_cat=10):
    np.random.seed(seed)

    classes_idx = np.random.choice(data.shape[0], size=[cats], replace=False)

    query_idx = np.random.randint(low=0, high=cats, size=())
    query_class = classes_idx[query_idx]

    features_mean = np.zeros((cats * examples_per_cat + 1, 512))

    y = []
    cur_label = 0
    idx = 0
    for i in range(cats):
        label = classes_idx[i]

        if i == query_idx:
            idxs = np.random.choice(data_len[label], size=[examples_per_cat + 1], replace=False)
            query_data = data[label, idxs[-1], :]
        else:
            idxs = np.random.choice(data_len[label], size=[examples_per_cat], replace=False)

        for j in range(examples_per_cat):
            features_mean[idx, :] = data[label, idxs[j], :]
            y.append(cur_label)
            idx += 1

        cur_label += 1

    features_mean[-1, :] = query_data
    y.append(query_idx)
    y = np.array(y)

    return features_mean, y


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def create_img_data(rng, data, data_len, params_e_size, cats=5, examples_per_cat=10, min_len=50):
    rng, new_rng, new_rng2, new_rng3 = jax.random.split(rng, num=4)

    # print("Features shape: ", data.shape)
    # print("Feature lengths shape: ", data_len.shape)

    classes_idx = jax.random.choice(new_rng, data.shape[0], shape=[cats], replace=False)
    classes_order = jax.random.permutation(new_rng2, cats)

    query_idx = classes_order[-1]
    query_class = classes_idx[query_idx]

    features_mean = jnp.zeros((cats * examples_per_cat + 1, 512))
    y = jnp.zeros((cats * examples_per_cat + 1))
    idx = 0

    data = data[:, :min_len, :]
    # print(data.shape)
    for i in range(cats - 1):
        rng, data_rng = jax.random.split(rng, num=2)
        label = classes_idx[classes_order[i]]
        cur_label = classes_order[i]

        idxs = jax.random.choice(data_rng, data.shape[1], shape=[examples_per_cat], replace=False).astype('int32')

        for j in range(examples_per_cat):
            features_mean = features_mean.at[idx, :].set(data[label, idxs[j], :])
            y = y.at[idx].set(cur_label)
            idx += 1

    # query
    rng, data_rng = jax.random.split(rng, num=2)
    label = classes_idx[classes_order[-1]]
    cur_label = classes_order[-1]
    query_idxs = jax.random.choice(data_rng, data.shape[1], shape=[examples_per_cat + 1], replace=False).astype('int32')
    for j in range(examples_per_cat):
        features_mean = features_mean.at[idx, :].set(data[label, query_idxs[j], :])
        y = y.at[idx].set(cur_label)
        idx += 1

    # print(y.shape)
    # print(y)

    features_mean_y = jnp.concatenate([features_mean, jnp.expand_dims(y, axis=-1)], axis=-1)
    features_mean_y_shuffled = jax.random.permutation(new_rng3, features_mean_y[:-1, :])
    # print(features_mean_y_shuffled.shape)

    features_mean = features_mean.at[:-1, :].set(features_mean_y_shuffled[:, :-1])

    # print(y[:-1].shape)
    # print(features_mean_y_shuffled[:, -1].shape)
    y = y.at[:-1].set(features_mean_y_shuffled[:, -1])

    features_mean = features_mean.at[-1, :].set(data[label, query_idxs[-1], :])
    y = y.at[-1].set(cur_label)
    y = y.squeeze()

    # print(y)

    x_query = features_mean[-1:, :]

    v_data_full = jax.nn.one_hot(y, cats)
    v_data = v_data_full[:, :-1] - 1 / cats * jnp.ones([1, cats - 1])

    v_target_full = v_data_full[-1:, :]
    v_target = v_data[-1:, :]

    v_data_full = v_data_full[:-1, :]

    query_init = jnp.zeros_like(v_target) - 1 / cats * jnp.ones([1, cats - 1])
    v_data = v_data.at[-1, :].set(jnp.squeeze(query_init))

    seq = jnp.concatenate([features_mean, jnp.concatenate((v_data_full, jnp.zeros((1, cats))), axis=0),
                           jnp.zeros((features_mean.shape[0], cats)) + 1 / cats,
                           jnp.zeros((features_mean.shape[0], params_e_size))],
                          axis=-1)
    target = jnp.concatenate([x_query, v_target_full, 1 / cats + jnp.zeros((x_query.shape[0], cats)),
                              jnp.zeros((x_query.shape[0], params_e_size))], axis=-1)

    return jnp.squeeze(seq), jnp.squeeze(target), jnp.squeeze(v_data_full), jnp.squeeze(v_target_full)


data_creator = vmap(create_img_data,
                    in_axes=(0, None, None, None, None, None, None), out_axes=0)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def create_img_data_linear(rng, data, data_len, params_e_size, cats=5, examples_per_cat=10, min_len=50):
    rng, new_rng, new_rng2, new_rng3 = jax.random.split(rng, num=4)

    # print("Features shape: ", data.shape)
    # print("Feature lengths shape: ", data_len.shape)

    classes_idx = jax.random.choice(new_rng, data.shape[0], shape=[cats], replace=False)
    classes_order = jax.random.permutation(new_rng2, cats)

    query_idx = classes_order[-1]
    query_class = classes_idx[query_idx]

    features_mean = jnp.zeros((cats * examples_per_cat + 1, 512))
    y = jnp.zeros((cats * examples_per_cat + 1))
    idx = 0

    data = data[:, :min_len, :]
    # print(data.shape)
    for i in range(cats - 1):
        rng, data_rng = jax.random.split(rng, num=2)
        label = classes_idx[classes_order[i]]
        cur_label = classes_order[i]

        idxs = jax.random.choice(data_rng, data.shape[1], shape=[examples_per_cat], replace=False).astype('int32')

        for j in range(examples_per_cat):
            features_mean = features_mean.at[idx, :].set(data[label, idxs[j], :])
            y = y.at[idx].set(cur_label)
            idx += 1

    # query
    rng, data_rng = jax.random.split(rng, num=2)
    label = classes_idx[classes_order[-1]]
    cur_label = classes_order[-1]
    query_idxs = jax.random.choice(data_rng, data.shape[1], shape=[examples_per_cat + 1], replace=False).astype('int32')
    for j in range(examples_per_cat):
        features_mean = features_mean.at[idx, :].set(data[label, query_idxs[j], :])
        y = y.at[idx].set(cur_label)
        idx += 1

    # print(y.shape)
    # print(y)

    features_mean_y = jnp.concatenate([features_mean, jnp.expand_dims(y, axis=-1)], axis=-1)
    features_mean_y_shuffled = jax.random.permutation(new_rng3, features_mean_y[:-1, :])
    # print(features_mean_y_shuffled.shape)

    features_mean = features_mean.at[:-1, :].set(features_mean_y_shuffled[:, :-1])

    # print(y[:-1].shape)
    # print(features_mean_y_shuffled[:, -1].shape)
    y = y.at[:-1].set(features_mean_y_shuffled[:, -1])

    features_mean = features_mean.at[-1, :].set(data[label, query_idxs[-1], :])
    y = y.at[-1].set(cur_label)
    y = y.squeeze()

    # print(y)

    x_query = features_mean[-1:, :]

    v_data_full = jax.nn.one_hot(y, cats)
    v_data = v_data_full[:, :-1] - 1 / cats * jnp.ones([1, cats - 1])

    v_target_full = v_data_full[-1:, :]
    v_target = v_data[-1:, :]

    seq = features_mean
    target = x_query

    return jnp.squeeze(seq), jnp.squeeze(target), jnp.squeeze(v_data_full)


data_creator = vmap(create_img_data_linear,
                    in_axes=(0, None, None, None, None, None, None), out_axes=0)


def create_weights(i_size, e_size, c_size, cats, lr, w_init, gd_deq=False, num_layers=1, use_bias_head=False,
                   gd_plus=False, widening_factor=4, one_hot_emb=False, linear_approx=False,
                   use_mlp=False, rng=None):
    """Create gradient descent weights for self-attention layers."""
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    # update f, head 1
    # query and key matrix
    query_upper_f_1 = jnp.concatenate([jnp.identity(i_size), jnp.zeros([i_size, 3 * e_size])], axis=1)
    query_lower_f_1 = jnp.zeros([3 * e_size, i_size + 3 * e_size])
    query_matrix_f_1 = jnp.concatenate([query_upper_f_1, query_lower_f_1], axis=0)
    key_matrix_f_1 = query_matrix_f_1

    # value matrix
    alpha = jax.random.normal(rng1, shape=(e_size,)) * config.init_scale  # randomly initialize alphas for learning rate
    value_left_f_1 = jnp.zeros([i_size + 3 * e_size, i_size + 2 * e_size])
    value_right_f_1 = jnp.concatenate([jnp.zeros([i_size, e_size]), jnp.diag(alpha), -1 * jnp.diag(alpha),
                                       jnp.zeros([e_size, e_size])], axis=0)
    value_matrix_f_1 = jnp.concatenate([value_left_f_1, value_right_f_1], axis=1) * 1 / c_size

    # projection matrix
    projection_matrix_f_1 = jnp.concatenate([jnp.zeros((i_size + 2 * e_size, i_size + 3 * e_size)),
                                             jnp.concatenate([jnp.zeros((e_size, i_size + 2 * e_size)),
                                                              jnp.identity(e_size)], axis=1)], axis=0)

    # update f, head 2
    # query and key matrix
    query_matrix_f_2 = config.lam * query_matrix_f_1
    key_matrix_f_2 = query_matrix_f_2

    # value matrix
    value_matrix_f_2 = jnp.identity(i_size + 3 * e_size)
    value_matrix_f_2 = value_matrix_f_2.at[:(i_size + e_size), :(i_size + e_size)].set(0)
    value_matrix_f_2 = value_matrix_f_2.at[-e_size:, -e_size:].set(0)

    # projection matrix
    projection_matrix_f_2 = -1 * jnp.identity(i_size + 3 * e_size)
    projection_matrix_f_2 = projection_matrix_f_2.at[:(i_size + e_size), :(i_size + e_size)].set(0)
    projection_matrix_f_2 = projection_matrix_f_2.at[-e_size:, -e_size:].set(0)

    # update f, both heads
    query_matrix_f = jnp.concatenate([query_matrix_f_1, query_matrix_f_2], axis=1)
    key_matrix_f = jnp.concatenate([key_matrix_f_1, key_matrix_f_2], axis=1)
    value_matrix_f = jnp.concatenate([value_matrix_f_1, value_matrix_f_2], axis=1)
    projection_matrix_f = jnp.concatenate([projection_matrix_f_1, projection_matrix_f_2], axis=0)

    # update e
    query_matrix_e = jnp.identity(i_size + 3 * e_size)
    query_matrix_e = query_matrix_e.at[:(i_size + 2 * e_size), :(i_size + 2 * e_size)].set(0)

    key_matrix_e = jnp.concatenate([jnp.zeros((e_size, i_size + 2 * e_size)), jnp.identity(e_size)], axis=1)
    value_matrix_e = jnp.concatenate([jnp.zeros((e_size, i_size + e_size)), jnp.identity(e_size),
                                      jnp.zeros((e_size, e_size))], axis=1)

    projection_matrix_e = jnp.identity(i_size + 3 * e_size)
    projection_matrix_e = projection_matrix_e.at[:(i_size + e_size), :(i_size + e_size)].set(0)
    projection_matrix_e = projection_matrix_e.at[-e_size:, -e_size:].set(0)

    # embedding matrix for attention
    if one_hot_emb:
        w_embedding = jnp.identity(e_size, dtype='float32')
        w_embedding = jnp.concatenate([w_embedding, jnp.zeros([1, e_size], dtype='float32')], axis=0)
    else:
        w_embedding = jax.random.normal(rng2, shape=[cats - 1, e_size]) * config.init_scale
        # w_embedding = jax.random.uniform(new_rng2, shape=[cats-1, e_size], minval=-config.init_scale, maxval=config.init_scale)
        w_embedding = jnp.concatenate([w_embedding, jnp.zeros([1, e_size])], axis=0)

    w_s_upper = jnp.concatenate([jnp.identity(i_size), jnp.zeros([i_size, 3 * e_size])], axis=1)
    w_s_middle_up = jnp.concatenate([jnp.zeros([cats, i_size]), w_embedding, jnp.zeros([cats, 2*e_size])], axis=1)
    w_s_middle_low = jnp.concatenate([jnp.zeros([cats, i_size + e_size]), w_embedding, jnp.zeros([cats, e_size])], axis=1)
    w_s_lower = jnp.concatenate([jnp.zeros([e_size, i_size + 2 * e_size]), jnp.identity(e_size)], axis=1)
    w_s = jnp.concatenate([w_s_upper, w_s_middle_up, w_s_middle_low, w_s_lower], axis=0)

    print("Embedding matrix: ", w_s)

    # Save params
    params_new = {}

    for l in range(num_layers):
        if num_layers == 1 or gd_deq:
            tra_name = 'Transformer_gd/update_'
        else:
            tra_name = 'Transformer_gd/~trans_block/layer_{}_'.format(l)

        params_new['Transformer_gd/emb'] = {'w': jnp.array(w_s)}

        params_new[tra_name + 'inner/query'] = {'w': jnp.array(query_matrix_f)}
        params_new[tra_name + 'inner/value'] = {'w': jnp.array(value_matrix_f)}
        params_new[tra_name + 'inner/key'] = {'w': jnp.array(key_matrix_f)}
        params_new[tra_name + 'inner/linear'] = {'w': jnp.array(projection_matrix_f)}

        params_new[tra_name + 'outer/query'] = {'w': jnp.array(query_matrix_e)}
        params_new[tra_name + 'outer/value'] = {'w': jnp.array(value_matrix_e)}
        params_new[tra_name + 'outer/key'] = {'w': jnp.array(key_matrix_e)}
        params_new[tra_name + 'outer/linear'] = {'w': jnp.array(projection_matrix_e)}

        if use_mlp:
            w1 = jax.random.normal(rng3, shape=[e_size, widening_factor * e_size]) * jnp.sqrt(0.002 / e_size)
            b1 = jax.random.normal(rng3, shape=[widening_factor * e_size]) * 0

            w2 = jax.random.normal(rng4, shape=[widening_factor * e_size, e_size]) * jnp.sqrt(0.002 / e_size)
            b2 = jax.random.normal(rng4, shape=[e_size]) * 0

            if gd_deq:
                params_new['Transformer_gd/mlp/linear'] = {'w': w1, 'b': b1}
                params_new['Transformer_gd/mlp/linear_1'] = {'w': w2, 'b': b2}
            else:
                params_new['Transformer_gd/~trans_block/layer_{}_mlp/linear'.format(l)] = {'w': w1, 'b': b1}
                params_new['Transformer_gd/~trans_block/layer_{}_mlp/linear_1'.format(l)] = {'w': w2, 'b': b2}

    return params_new


rng = jax.random.PRNGKey(1)
create_weights(2, 5, 100, 3, 0.1, jnp.ones([1]) * 0.1, gd_deq=False, num_layers=1, use_bias_head=False,
               gd_plus=False, widening_factor=4, use_mlp=True, rng=rng)
