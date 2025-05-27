"""Training fleixble Transformer model.

"""
import sys
from functools import partial
from typing import Any, MutableMapping, NamedTuple, Tuple

from absl import app

import numpy as np
import haiku as hk
import jax
from jax import jit, vmap
import jax.numpy as jnp
import optax

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from IPython.display import display

from transformer import Transformer
from data import create_cat_data_grid, create_cat_data_random_grid, create_cat_data_high_dim, \
    create_img_data, create_weights, create_w_e
from config import config

from datetime import datetime
import pytz

cet = pytz.timezone('CET')

file_time = str(datetime.now(tz=cet))

data_creator = vmap(create_cat_data_random_grid,
                    in_axes=(0, None, None, None, None, None, None, None, None), out_axes=0)

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'


class TrainState(NamedTuple):
    """Container for the training state."""
    params: hk.Params
    opt_state: optax.OptState
    rng: jax.Array
    step: jax.Array


class TestState(NamedTuple):
    """Container for the test state."""
    prediction: jax.Array
    inter_losses: jax.Array
    test_loss: jax.Array
    rng: jax.Array
    step: jax.Array


class DataState(NamedTuple):
    """Container for the data state."""
    train_data: jax.Array
    test_data: jax.Array
    rng: jax.Array
    step: jax.Array


_Metrics = MutableMapping[str, Any]


def change_dataloader():
    """Set dataloader."""
    global data_creator

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


def forward(tokens: jnp.ndarray, is_training: bool, gd: bool):
    """Transformer forward pass."""
    tr = Transformer(
        num_heads=config.num_heads_tr,
        num_layers=config.num_layers,
        key_size=config.key_size,
        input_size=config.input_size,
        output_size=config.cats,
        emb_size=config.e_size,
        widening_factor=config.widening_factor,
        mlp_output_dim=config.mlp_output_dim,
        second_layer=config.second_layer,
        input_mapping=config.in_proj,
        use_bias_p=config.use_bias,
        deq=config.deq,
        init_scale=config.init_scale,
        sum_norm=config.sum_norm,
        ana_copy=config.ana_copy,
        kernel=config.model_kernel,
        gamma=config.gamma,
        sigma=config.sigma,
        num_queries=config.num_queries,
        linear_approx=config.linear_approx,
        use_mlp=config.use_mlp)

    tr_gd = Transformer(
        num_heads=config.num_heads_gd,
        num_layers=config.num_layers,
        key_size=config.key_size,
        input_size=config.input_size,
        output_size=config.cats,
        emb_size=config.e_size,
        widening_factor=config.widening_factor,
        mlp_output_dim=config.mlp_output_dim,
        second_layer=config.second_layer,
        input_mapping=config.in_proj,
        use_bias_p=False,
        deq=config.gd_deq,
        sum_norm=False,
        init_scale=config.init_scale,
        kernel=config.model_kernel,
        gamma=config.gamma,
        sigma=config.sigma,
        num_queries=config.num_queries,
        linear_approx=config.linear_approx,
        use_mlp=config.use_mlp,
        name='Transformer_gd')

    if not gd:
        return tr(tokens, is_training=is_training, predict_test=False)
    else:
        return tr_gd(tokens, is_training=is_training, predict_test=False)


def grid_to_reg(data):
    """Reshapes data created with the "grid" kernel so that it can be used with the Transformer."""
    data = list(data)

    data[0] = jnp.reshape(data[0], (-1, data[0].shape[2], data[0].shape[3]))  # input data sequence
    data[1] = jnp.reshape(data[1], (-1, data[1].shape[2]))  # target
    data[2] = jnp.reshape(data[2], (-1, data[2].shape[2], data[2].shape[3]))  # prob
    data[3] = jnp.reshape(data[3], (-1, data[3].shape[2]))  # prob target
    data[4] = jnp.reshape(data[4], (-1, data[4].shape[2]))  # v_data
    data[5] = jnp.reshape(data[5], (-1, data[5].shape[2]))  # v_target
    data[6] = jnp.reshape(data[6], (-1, data[6].shape[2], data[6].shape[3]))  # W_e
    data[7] = jnp.reshape(data[7], (-1, data[7].shape[2]))  # v_target_full

    data = tuple(data)
    return data


def get_mean_probs(data):
    """Computes the mean maximum probability and the mean target probability of a dataset."""
    y_targets = data[-1]
    prob_targets = data[3].squeeze()
    prob_targets_iso = jnp.where(y_targets == 1, prob_targets, 0).sum(axis=1)

    mean_max_prob = jnp.max(prob_targets, axis=1).mean()
    mean_target_prob = jnp.mean(prob_targets_iso)

    return mean_max_prob, mean_target_prob


def compute_loss(preds, targets):
    """Computes the negative log-likelihood (NLL) loss between predictions and targets."""
    assert preds.shape[0] == targets.shape[0]

    # negative log-likelihood
    loss = jnp.mean(-1 * jnp.log(jnp.sum(jnp.where(targets == 1, preds + 1e-20, 0), axis=1)))

    return loss


def compute_mse_loss(preds, targets):
    """Computes the mean squared error (MSE) loss between predictions and targets."""
    assert preds.shape == targets.shape

    # mse loss
    loss = 0.5 * jnp.sum((targets - preds) ** 2) / targets.shape[0]

    return loss


@hk.transform
def loss_fn(data: jnp.ndarray, gd) -> jnp.ndarray:
    """Computes the NLL loss between targets and predictions."""
    preds, _, _ = forward(data[0], True, gd)
    y_targets = data[-1]

    return compute_loss(preds, y_targets)


@hk.transform
def predict(data: jnp.ndarray, gd) -> Tuple[jnp.ndarray]:
    """Predict."""
    preds, _, _ = forward(data, False, gd)

    return preds


@hk.transform
def predict_stack(data: jnp.ndarray, gd) -> Tuple[jnp.ndarray]:
    """Predict and return stack."""
    _, stack, _ = forward(data, False, gd)

    return stack


@hk.transform
def predict_attn(data: jnp.ndarray, gd) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Predict and return stack."""
    _, _, attn = forward(data, False, gd)

    return attn


@hk.transform
def predict_test(data: jnp.ndarray, gd) -> Tuple[jnp.ndarray, jnp.ndarray,
jnp.ndarray, jnp.ndarray]:
    """Predict test data used for analyses as well as metrics computation."""
    preds, pred_stack, _ = forward(data[0], False, gd)

    pred_stack = jnp.array(pred_stack)

    y_targets = data[-1]

    probs = preds

    # compute loss
    loss_final = compute_loss(probs, y_targets)

    loss_f = lambda x: compute_loss(x, y_targets)

    if not config.ana_copy:
        losses = vmap(loss_f)(pred_stack)
    else:
        losses = []

    # compute accuracy
    preds_cat = jnp.argmax(probs, axis=-1).squeeze()
    targets_cat = jnp.argmax(y_targets, axis=-1).squeeze()

    acc = jnp.where(preds_cat == targets_cat, 1, 0).sum(axis=None) / targets_cat.shape[0]

    # compute top-3 frequency
    top_3_preds = jnp.argsort(probs, axis=-1)[:, -config.top_preds:]
    top_3_freq = jnp.where(targets_cat[:, None] == top_3_preds, 1, 0).sum(axis=None) / targets_cat.shape[0]

    # probability distance
    if config.data_kernel != 'imagenet':
        prob_targets = data[3].squeeze()
        prob_preds_iso = jnp.where(y_targets == 1, probs, 0).sum(axis=1)
        prob_targets_iso = jnp.where(y_targets == 1, prob_targets, 0).sum(axis=1)

        prob_dist = compute_mse_loss(prob_preds_iso, prob_targets_iso)
    else:
        prob_dist = None

    return loss_final, pred_stack, losses, acc, top_3_freq, prob_dist


@partial(jax.jit, static_argnums=(2, 3, 4))
def update(state: TrainState, data, optimizer, gd=False, constructed=True) -> Tuple[TrainState, _Metrics]:
    """Does an SGD step and returns training state as well as metrics."""
    rng, new_rng = jax.random.split(state.rng)
    jit_loss_apply = jit(loss_fn.apply, static_argnums=3)

    loss_and_grad_fn = jax.value_and_grad(jit_loss_apply)
    loss, gradients = loss_and_grad_fn(state.params, rng, data, gd)

    # set embedding matrix for Trained TF
    if not constructed:
        if config.one_hot_emb:
            gradients['transformer/emb']['w'] = jnp.zeros_like(gradients['transformer/emb']['w'])
        else:
            emb_cutoff = np.ones(
                (config.input_size + 2*config.cats + config.e_size, config.input_size + 3 * config.e_size),
                dtype=bool)
            emb_cutoff[config.input_size:(config.input_size + config.cats), config.input_size:config.input_size + config.e_size] = False
            emb_cutoff[(config.input_size + config.cats):(config.input_size + 2*config.cats), config.input_size + config.e_size:-config.e_size] = False
            gradients['transformer/emb']['w'] = gradients['transformer/emb']['w'].at[emb_cutoff].set(0)

    # update parameters with optimizer step
    updates, new_opt_state = optimizer.update(gradients, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainState(
        params=new_params,
        opt_state=new_opt_state,
        rng=new_rng,
        step=state.step + 1,
    )

    metrics = {
        'step': state.step,
        'train_loss': loss,
    }

    return new_state, metrics


@jax.jit
def evaluation(train_state: TrainState,
               test_state: TestState, data, gd) -> TestState:
    """Compute predictions from model."""
    rng, new_rng = jax.random.split(test_state.rng)
    loss, preds, inter_losses, _, _, _ = predict_test.apply(train_state.params, rng, data, gd)
    new_state = TestState(
        prediction=preds,
        inter_losses=inter_losses,
        test_loss=loss,
        rng=new_rng,
        step=test_state.step + 1,
    )

    return new_state


def init_model(rng, train_data, test_data, optimizer, constructed) -> TrainState:
    """Initialize haiku transform modules to create train and test state."""
    train_rng, test_rng = jax.random.split(rng, num=2)
    initial_params = loss_fn.init(rng, train_data, gd=False)

    # set embedding matrix for Trained TF
    if not constructed:
        w_s_upper = jnp.concatenate([jnp.identity(config.input_size), jnp.zeros([config.input_size,
                                                                                 3 * config.e_size])], axis=1)
        w_s_lower = jnp.concatenate([jnp.zeros((config.e_size, config.input_size + 2 * config.e_size)),
                                     jnp.identity(config.e_size)], axis=1)
        initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[:config.input_size, :].set(
            w_s_upper)
        initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[
                                                 config.input_size:config.input_size + 2*config.cats, :config.input_size].set(
            jnp.zeros([2*config.cats, config.input_size]))
        initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[
                                                 config.input_size + config.cats:config.input_size + 2 * config.cats,
                                                 config.input_size:config.input_size + config.e_size].set(
            jnp.zeros([config.cats, config.e_size]))
        initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[
                                                 config.input_size:config.input_size + config.cats,
                                                 config.input_size + config.e_size:config.input_size + 2*config.e_size].set(
            jnp.zeros([config.cats, config.e_size]))
        initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[
                                                 config.input_size:config.input_size + 2 * config.cats,
                                                 -config.e_size:].set(
            jnp.zeros([2 * config.cats, config.e_size]))
        initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[-config.e_size:, :].set(
            w_s_lower)

        if config.one_hot_emb:
            initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[
                                                     config.input_size:config.input_size + config.cats,
                                                     config.input_size:(config.input_size + config.e_size)].set(
                jnp.concatenate([
                    jnp.identity(config.e_size, dtype='float32'),
                    jnp.zeros([1, config.e_size], dtype='float32')], axis=0))

            initial_params['transformer/emb']['w'] = initial_params['transformer/emb']['w'].at[
                                                     config.input_size + config.cats:config.input_size + 2 * config.cats,
                                                     (config.input_size + config.e_size):-config.e_size].set(
                jnp.concatenate([
                    jnp.identity(config.e_size, dtype='float32'),
                    jnp.zeros([1, config.e_size], dtype='float32')], axis=0))

        print("emb shape: ", initial_params['transformer/emb']['w'].shape)

    if config.analyze:
        initial_params_gd = loss_fn.init(rng, train_data, gd=True)
        _, _, _, _, _, _, = predict_test.apply(initial_params_gd, rng, test_data, True)

    initial_test_loss, initial_preds, i_inter_losses, _, _, _ = predict_test.apply(
        initial_params,
        rng,
        test_data, False)
    _ = predict.apply(initial_params, rng, test_data[0], False)
    _ = predict_stack.apply(initial_params, rng, test_data[0], False)

    initial_opt_state = optimizer.init(initial_params)

    return TrainState(
        params=initial_params,
        opt_state=initial_opt_state,
        rng=train_rng,
        step=np.array(0)), TestState(
        prediction=initial_preds,
        inter_losses=i_inter_losses,
        test_loss=initial_test_loss,
        rng=test_rng,
        step=np.array(0))


def init(constructed):
    """Initialize data creator, model, optimizer, etc."""
    rng = jax.random.PRNGKey(config.seed)
    rng, train_rng = jax.random.split(rng, 2)

    if config.data_kernel == 'grid':
        train_data = data_creator(jax.random.split(train_rng, num=config.mb_size),
                                  config.input_size,
                                  config.dataset_size,
                                  config.e_size,
                                  config.cats,
                                  config.input_range,
                                  config.weight_scale,
                                  config.bias_data)

        train_data = grid_to_reg(train_data)
    elif config.data_kernel == 'random_grid':
        W_e = create_w_e(train_rng, config.e_size, config.cats)

        train_data = data_creator(jax.random.split(train_rng, num=config.mb_size),
                                  config.input_size,
                                  config.dataset_size,
                                  config.e_size,
                                  config.cats,
                                  config.input_range,
                                  config.weight_scale,
                                  None if config.unique_w_e else W_e,
                                  config.bias_data)
    elif config.data_kernel == 'high_dim':
        W_e = create_w_e(train_rng, config.data_e_size, config.cats)

        sharding = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))

        train_data = data_creator(
            jax.device_put(jax.random.split(train_rng, num=config.mb_size), sharding.reshape(2, 1)),
            config.input_size,
            config.dataset_size,
            config.e_size,
            config.cats,
            config.k,
            config.dist,
            config.l,
            None if config.unique_w_e else W_e,
            config.input_range)

        train_data = list(train_data)
        train_data[0] = jax.device_put(train_data[0], sharding.reshape(2, 1, 1))
        train_data = tuple(train_data)
    elif config.data_kernel == 'imagenet':
        sharding = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))

        train_data = data_creator(
            jax.device_put(jax.random.split(train_rng, num=config.mb_size), sharding.reshape(2, 1)),
            config.train_features,
            config.train_lens,
            config.e_size,
            config.cats,
            config.examples_per_cat,
            config.train_min_len)

        train_data = list(train_data)
        train_data[0] = jax.device_put(train_data[0], sharding.reshape(2, 1, 1))
        train_data = tuple(train_data)

    if config.tf_use_lr_schedule:
        lr = optax.exponential_decay(init_value=config.lr, transition_steps=config.tf_transition_steps,
                                     decay_rate=config.lr_decay_rate, transition_begin=config.transition_begin,
                                     staircase=config.staircase)
    else:
        lr = config.lr
    if config.adam:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_value),
            optax.adamw(learning_rate=lr, b1=config.b1, b2=config.b2,
                        weight_decay=config.wd),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_value),
            optax.sgd(learning_rate=lr, ),
        )

    train_state, test_state = init_model(rng, train_data, train_data, optimizer, constructed)

    return optimizer, train_state, test_state, rng


@jax.jit
def analyze(data, state, rng, params_constructed):
    """Analyze alignment between GD and trained Transformer."""
    # Trained Transformer
    pred = lambda z: predict.apply(state.params, rng,
                                   z[None, ...], False)[0, -1, -config.output_size:]
    predictions = vmap(pred)(data[0])

    # GD
    pred_c = lambda z: predict.apply(params_constructed,
                                     rng, z[None, ...], True)[0, -1, -config.output_size:]
    predictions_c = vmap(pred_c)(data[0])

    dot = None
    norm = None

    pred_norm = jnp.mean(jnp.linalg.norm(predictions[..., None] -
                                         predictions_c[..., None], axis=1))

    return dot, norm, pred_norm


@partial(jax.jit, static_argnums=(1, 2))
def gradient_manipulation_classification_constructed(gradients, ndim, linear=False):
    """Manipulates gradients of gradient descent for GD model."""
    update_matrix = np.eye(ndim, dtype=bool)
    indx = np.where(~update_matrix)
    key_cutoff = np.eye(ndim, dtype=bool)
    key_cutoff[:config.input_size, :config.input_size] = False

    aug_gradients = {}

    grid_key_query_manipulated = False
    learned_key_query_param = None
    learned_m = None

    for param in gradients:
        if 'mlp' in param:
            # MLP element: don't change gradients
            aug_gradients[param] = gradients[param]
        elif 'linear' in param:
            # projection matrix: set all gradients to 0
            gradients[param]['w'] = jnp.zeros_like(gradients[param]['w'])

            aug_gradients[param] = gradients[param]
        elif not linear and ('query' in param or 'key' in param): # non-linear model kernel, key/query matrix
            if 'inner' in param:
                # inner layer of cross-attention
                if not config.gd_plus:
                    key_select = np.concatenate((np.eye(ndim, dtype=bool), np.zeros((ndim, ndim), dtype=bool)), axis=1)
                    key_select[config.input_size:, config.input_size:] = False

                    gradients[param]['w'] = gradients[param]['w'].at[~key_select].set(0)
                    if config.gd_deq:
                        if not grid_key_query_manipulated:
                            learned_key_query_param = jnp.mean(gradients[param]['w'][key_select])

                            grid_key_query_manipulated = True

                        gradients[param]['w'] = gradients[param]['w'].at[key_select].set(learned_key_query_param)
                    else:
                        for l in range(config.num_layers):
                            if str(l) in param:
                                if not grid_key_query_manipulated:
                                    learned_key_query_param = jnp.mean(gradients[param]['w'][key_select])

                                    grid_key_query_manipulated = True

                                gradients[param]['w'] = gradients[param]['w'].at[key_select].set(
                                    learned_key_query_param)
            elif 'outer' in param:
                # outer layer of cross-attention: set all gradients to 0
                gradients[param]['w'] = jnp.zeros_like(gradients[param]['w'])

            aug_gradients[param] = gradients[param]
        elif "value" in param: # value matrix
            if 'inner' in param:
                # inner layer of cross-attention
                value_cutoff_right = np.concatenate([np.zeros((config.input_size, config.e_size), dtype=bool),
                                                     np.identity(config.e_size, dtype=bool),
                                                     np.identity(config.e_size, dtype=bool),
                                                     np.zeros((config.e_size, config.e_size), dtype=bool)], axis=0)
                value_cutoff = ~np.concatenate([np.zeros((config.input_size + 3 * config.e_size,
                                                          config.input_size + 2 * config.e_size), dtype=bool),
                                                value_cutoff_right,
                                                np.zeros((config.input_size + 3 * config.e_size,
                                                          config.input_size + 3 * config.e_size), dtype=bool)], axis=1)

                gradients[param]['w'] = gradients[param]['w'].at[value_cutoff].set(0)

                cutoff_1 = gradients[param]['w'][config.input_size:config.input_size + config.e_size,
                           (config.input_size + 2 * config.e_size):(config.input_size + 3 * config.e_size)][
                    np.identity(config.e_size, dtype=bool)]
                cutoff_2 = gradients[param]['w'][config.input_size + config.e_size:config.input_size + 2*config.e_size,
                           (config.input_size + 2 * config.e_size):(config.input_size + 3 * config.e_size)][
                    np.identity(config.e_size, dtype=bool)]

                gradients[param]['w'] = gradients[param]['w'].at[config.input_size:config.input_size + config.e_size,
                                        (config.input_size + 2 * config.e_size):(
                                                config.input_size + 3 * config.e_size)].set(
                    jnp.diag(cutoff_1))
                gradients[param]['w'] = gradients[param]['w'].at[config.input_size + config.e_size:config.input_size
                                                                                                   + 2*config.e_size,
                                        (config.input_size + 2 * config.e_size):(
                                                config.input_size + 3 * config.e_size)].set(
                    jnp.diag(-1 * cutoff_1))
            elif 'outer' in param:
                # outer layer of cross-attention: set all elements to 0
                gradients[param]['w'] = jnp.zeros_like(gradients[param]['w'])

            aug_gradients[param] = gradients[param]
        elif "emb" in param:  # embedding matrix
            if config.one_hot_emb:
                gradients[param]['w'] = jnp.zeros_like(gradients[param]['w'])
            else:
                cutoff_1 = gradients[param]['w'][config.input_size:config.input_size + config.cats,
                           (config.input_size):(config.input_size + config.e_size)]
                cutoff_2 = gradients[param]['w'][config.input_size + config.cats:config.input_size + 2 * config.cats,
                           (config.input_size + config.e_size):(config.input_size + 2 * config.e_size)]
                gradients[param]['w'] = gradients[param]['w'].at[config.input_size:config.input_size + config.cats,
                                        (config.input_size):(config.input_size + config.e_size)].set(cutoff_1)
                gradients[param]['w'] = gradients[param]['w'].at[
                                        config.input_size + config.cats:config.input_size + 2 * config.cats,
                                        (config.input_size + config.e_size):(
                                                config.input_size + 2 * config.e_size)].set(cutoff_1)

                # only update weight embeddings, and leave rest of embedding matrix constant
                emb_cutoff = np.ones(
                    (config.input_size + 2 * config.cats + config.e_size, config.input_size + 3 * config.e_size),
                    dtype=bool)

                # only update embedding vectors c=1,...,C-1 (not C)
                emb_cutoff[config.input_size:(config.input_size + config.cats - 1),
                config.input_size:(config.input_size + config.e_size)] = False
                emb_cutoff[config.input_size + config.cats:(config.input_size + 2 * config.cats - 1),
                (config.input_size + config.e_size):(config.input_size + 2 * config.e_size)] = False

                gradients[param]['w'] = gradients[param]['w'].at[emb_cutoff].set(0)

            aug_gradients[param] = gradients[param]
        else:
            # any other matrix: set all gradients to zero
            aug_gradients[param] = {'w': jnp.zeros_like(gradients[param]['w'])}

    return aug_gradients


def pre_train_gd_classification(train_data_rng, val_rng, eval_rng, w_e, params_gd, linear):
    """Train GD transformer."""
    # initialize optimizer
    if config.gd_use_lr_schedule:
        gd_lr = optax.exponential_decay(init_value=config.gd_lr, transition_steps=config.gd_transition_steps,
                                        decay_rate=config.gd_lr_decay_rate, transition_begin=config.transition_begin,
                                        staircase=config.staircase)
    else:
        gd_lr = config.gd_lr

    optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip_value_gd),
                            optax.adam(gd_lr, b1=0.9, b2=0.999))
    opt_state = optimizer.init(params_gd)

    # create evaluation data
    if config.data_kernel == 'grid':
        eval_data = data_creator(jax.random.split(eval_rng, num=config.bs_eval),
                                 config.input_size,
                                 config.dataset_size,
                                 config.e_size,
                                 config.cats,
                                 config.input_range,
                                 config.weight_scale,
                                 config.bias_data)
        eval_data = grid_to_reg(eval_data)

        val_data = data_creator(jax.random.split(val_rng, num=config.bs_eval),
                                config.input_size,
                                config.dataset_size,
                                config.e_size,
                                config.cats,
                                config.input_range,
                                config.weight_scale,
                                config.bias_data)

        val_data = grid_to_reg(val_data)
    elif config.data_kernel == 'random_grid':
        eval_data = data_creator(jax.random.split(eval_rng, num=config.bs_eval),
                                 config.input_size,
                                 config.dataset_size,
                                 config.e_size,
                                 config.cats,
                                 config.input_range,
                                 config.weight_scale,
                                 None if config.unique_w_e else w_e,
                                 config.bias_data)

        val_data = data_creator(jax.random.split(val_rng, num=config.bs_eval),
                                config.input_size,
                                config.dataset_size,
                                config.e_size,
                                config.cats,
                                config.input_range,
                                config.weight_scale,
                                None if config.unique_w_e else w_e,
                                config.bias_data)
    elif config.data_kernel == 'high_dim':
        sharding = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))

        eval_data = data_creator(jax.device_put(jax.random.split(eval_rng, num=config.bs_eval), sharding.reshape(2, 1)),
                                 config.input_size,
                                 config.dataset_size,
                                 config.e_size,
                                 config.cats,
                                 config.k,
                                 config.dist,
                                 config.l,
                                 None if config.unique_w_e else w_e,
                                 config.input_range)

        val_data = data_creator(jax.device_put(jax.random.split(val_rng, num=config.bs_eval), sharding.reshape(2, 1)),
                                config.input_size,
                                config.dataset_size,
                                config.e_size,
                                config.cats,
                                config.k,
                                config.dist,
                                config.l,
                                None if config.unique_w_e else w_e,
                                config.input_range)

        eval_data = list(eval_data)
        eval_data[0] = jax.device_put(eval_data[0], sharding.reshape(2, 1, 1))
        eval_data = tuple(eval_data)

        val_data = list(val_data)
        val_data[0] = jax.device_put(val_data[0], sharding.reshape(2, 1, 1))
        val_data = tuple(val_data)
    elif config.data_kernel == 'imagenet':
        sharding = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))

        eval_data = data_creator(jax.device_put(jax.random.split(eval_rng, num=config.bs_eval), sharding.reshape(2, 1)),
                                 config.val_features,
                                 config.val_lens,
                                 config.e_size,
                                 config.cats,
                                 config.examples_per_cat,
                                 config.val_min_len)

        val_data = data_creator(jax.device_put(jax.random.split(val_rng, num=config.bs_eval), sharding.reshape(2, 1)),
                                config.train_features,
                                config.train_lens,
                                config.e_size,
                                config.cats,
                                config.examples_per_cat,
                                config.train_min_len)

        eval_data = list(eval_data)
        eval_data[0] = jax.device_put(eval_data[0], sharding.reshape(2, 1, 1))
        eval_data = tuple(eval_data)

        val_data = list(val_data)
        val_data[0] = jax.device_put(val_data[0], sharding.reshape(2, 1, 1))
        val_data = tuple(val_data)

    if config.data_kernel == 'imagenet':
        eval_mean_max_prob = None
        eval_mean_target_prob = None
        val_mean_max_prob = None
        val_mean_target_prob = None
    else:
        eval_mean_max_prob, eval_mean_target_prob = get_mean_probs(eval_data)
        val_mean_max_prob, val_mean_target_prob = get_mean_probs(val_data)

    # initialize empty lists for metrics
    eval_gd_losses = []
    eval_gd_acc = []
    eval_gd_top_3_freq = []
    eval_gd_prob_dist = []
    eval_mean_max_prob_list = []
    eval_mean_target_prob_list = []

    val_gd_losses = []
    val_gd_acc = []
    val_gd_top_3_freq = []
    val_gd_prob_dist = []
    val_mean_max_prob_list = []
    val_mean_target_prob_list = []

    train_steps = config.training_steps_gd_constructed

    waiting_time = -1
    patience = 15
    best_params_gd = params_gd
    best_step = 0
    best_val_loss = jnp.inf
    best_val_acc = 0
    best_val_mse = jnp.inf
    best_val_top_3_freq = 0

    # if config.data_kernel == 'grid':
    #     train_data = data_creator(jax.random.split(train_data_rng, num=config.bs_train),
    #                               config.input_size,
    #                               config.dataset_size,
    #                               config.e_size,
    #                               config.cats,
    #                               config.input_range,
    #                               config.weight_scale,
    #                               config.bias_data)
    #     train_data = grid_to_reg(train_data)
    # elif config.data_kernel == 'random_grid':
    #     train_data = data_creator(jax.random.split(train_data_rng, num=config.bs_train),
    #                               config.input_size,
    #                               config.dataset_size,
    #                               config.e_size,
    #                               config.cats,
    #                               config.input_range,
    #                               config.weight_scale,
    #                               None if config.unique_w_e else w_e,
    #                               config.bias_data)
    # elif config.data_kernel == 'high_dim':
    #     train_data = data_creator(
    #         jax.device_put(jax.random.split(train_data_rng, num=config.bs_train), sharding.reshape(4, 1)),
    #         config.input_size,
    #         config.dataset_size,
    #         config.e_size,
    #         config.cats,
    #         config.k,
    #         config.dist,
    #         config.l,
    #         None if config.unique_w_e else w_e,
    #         config.input_range)
    #
    #     train_data = list(train_data)
    #     train_data[0] = jax.device_put(train_data[0], sharding.reshape(4, 1, 1))
    #     train_data = tuple(train_data)
    # elif config.data_kernel == 'imagenet':
    #     train_data = data_creator(
    #         jax.device_put(jax.random.split(train_data_rng, num=config.bs_train), sharding.reshape(4, 1)),
    #         config.train_features,
    #         config.train_lens,
    #         config.e_size,
    #         config.cats,
    #         config.examples_per_cat,
    #         config.train_min_len)
    #
    #     train_data = list(train_data)
    #     train_data[0] = jax.device_put(train_data[0], sharding.reshape(4, 1, 1))
    #     train_data = tuple(train_data)

    # rng, train_data_rng = jax.random.split(train_data_rng, 2)

    # train GD model
    original_data_rng = train_data_rng
    rng, _ = jax.random.split(train_data_rng, 2)
    for step in range(train_steps):
        if config.cycle_data > 0:
            if step % config.cycle_data == 0:
                train_data_rng = original_data_rng

        # generate training data
        # rng, train_data_rng = jax.random.split(train_data_rng, 2)
        if config.data_kernel == 'grid':
            train_data = data_creator(jax.random.split(train_data_rng, num=config.bs_train),
                                      config.input_size,
                                      config.dataset_size,
                                      config.e_size,
                                      config.cats,
                                      config.input_range,
                                      config.weight_scale,
                                      config.bias_data)
            train_data = grid_to_reg(train_data)
        elif config.data_kernel == 'random_grid':
            train_data = data_creator(jax.random.split(train_data_rng, num=config.bs_train),
                                      config.input_size,
                                      config.dataset_size,
                                      config.e_size,
                                      config.cats,
                                      config.input_range,
                                      config.weight_scale,
                                      None if config.unique_w_e else w_e,
                                      config.bias_data)
        elif config.data_kernel == 'high_dim':
            train_data = data_creator(
                jax.device_put(jax.random.split(train_data_rng, num=config.bs_train), sharding.reshape(2, 1)),
                config.input_size,
                config.dataset_size,
                config.e_size,
                config.cats,
                config.k,
                config.dist,
                config.l,
                None if config.unique_w_e else w_e,
                config.input_range)

            train_data = list(train_data)
            train_data[0] = jax.device_put(train_data[0], sharding.reshape(2, 1, 1))
            train_data = tuple(train_data)
        elif config.data_kernel == 'imagenet':
            train_data = data_creator(
                jax.device_put(jax.random.split(train_data_rng, num=config.bs_train), sharding.reshape(2, 1)),
                config.train_features,
                config.train_lens,
                config.e_size,
                config.cats,
                config.examples_per_cat,
                config.train_min_len)

            train_data = list(train_data)
            train_data[0] = jax.device_put(train_data[0], sharding.reshape(2, 1, 1))
            train_data = tuple(train_data)

        rng, _ = jax.random.split(rng, 2)
        data_idxs = jax.random.permutation(rng, len(train_data[0]))
        num_minibatches = len(data_idxs) // config.mb_size

        for i in range(num_minibatches):
            mini_batch = tuple(data[data_idxs[i * config.mb_size:(i + 1) * config.mb_size]] for data in train_data)

            jit_loss_apply = jit(loss_fn.apply, static_argnums=3)
            loss_and_grad_fn = jax.value_and_grad(jit_loss_apply)
            loss, gradients = loss_and_grad_fn(params_gd, rng, mini_batch, True)

            # with jnp.printoptions(threshold=sys.maxsize):
                # print("original gradients: ", gradients)
            # for key, value in gradients.items():
            #     print(key, ": ", np.where(value['w'] != 0, 1, 0).sum())
            # # manipulate gradients
            aug_gradients = gradient_manipulation_classification_constructed(gradients, config.key_size, linear)

            # with jnp.printoptions(threshold=sys.maxsize):
            #     print("aug gradients: ", aug_gradients)
            # print("aug gradients")
            # for key, value in aug_gradients.items():
            #     print(key, ": ", np.where(value['w'] != 0, 1, 0).sum())

            # update parameters
            updates, opt_state = optimizer.update(aug_gradients, opt_state)
            params_gd = optax.apply_updates(params_gd, updates)

        if len(data_idxs) % config.mb_size != 0:
            mini_batch = tuple(data[data_idxs[(i + 1) * config.mb_size:]] for data in train_data)

            jit_loss_apply = jit(loss_fn.apply, static_argnums=3)
            loss_and_grad_fn = jax.value_and_grad(jit_loss_apply)
            loss, gradients = loss_and_grad_fn(params_gd, rng, mini_batch, True)

            # manipulate gradients
            aug_gradients = gradient_manipulation_classification_constructed(gradients, config.key_size, linear)

            # update parameters
            updates, opt_state = optimizer.update(aug_gradients, opt_state)
            params_gd = optax.apply_updates(params_gd, updates)

        # jit_loss_apply = jit(loss_fn.apply, static_argnums=3)
        # loss_and_grad_fn = jax.value_and_grad(jit_loss_apply)
        # loss, gradients = loss_and_grad_fn(params_gd, train_data_rng, train_data, True)
        #
        # # manipulate gradients
        # aug_gradients = gradient_manipulation_classification_constructed(gradients, config.key_size, linear)
        #
        # # update parameters
        # updates, opt_state = optimizer.update(aug_gradients, opt_state)
        # params_gd = optax.apply_updates(params_gd, updates)

        # evaluate model every 100 steps
        if step % 100 == 0:
            if config.data_kernel == 'imagenet' and config.model_kernel=='rbf':
                data_idxs = jnp.arange(len(val_data[0]))
                num_minibatches = len(data_idxs) // config.mb_size

                val_losses_gd = 0
                val_acc_gd = 0
                val_top_3_freq_gd = 0
                val_prob_dist_gd = None if config.data_kernel == 'imagenet' else 0

                for i in range(num_minibatches):
                    mini_batch = tuple(data[data_idxs[i * config.mb_size:(i + 1) * config.mb_size]] for data in val_data)

                    val_losses_gd_temp, _, _, val_acc_gd_temp, val_top_3_freq_gd_temp, val_prob_dist_gd_temp = predict_test.apply(params_gd,
                                                                                                              val_rng, mini_batch,
                                                                                                              True)
                    val_losses_gd += val_losses_gd_temp
                    val_acc_gd += val_acc_gd_temp
                    val_top_3_freq_gd += val_top_3_freq_gd_temp
                    if config.data_kernel != 'imagenet':
                        val_prob_dist_gd += val_prob_dist_gd_temp

                val_losses_gd /= num_minibatches
                val_acc_gd /= num_minibatches
                val_top_3_freq_gd /= num_minibatches
                if config.data_kernel != 'imagenet':
                    val_prob_dist_gd /= num_minibatches

                eval_losses_gd = 0
                eval_acc_gd = 0
                eval_top_3_freq_gd = 0
                eval_prob_dist_gd = None if config.data_kernel == 'imagenet' else 0

                for i in range(num_minibatches):
                    mini_batch = tuple(data[data_idxs[i * config.mb_size:(i + 1) * config.mb_size]] for data in eval_data)

                    eval_losses_gd_temp, _, _, eval_acc_gd_temp, eval_top_3_freq_gd_temp, eval_prob_dist_gd_temp = predict_test.apply(
                        params_gd,
                        eval_rng, mini_batch,
                        True)
                    eval_losses_gd += eval_losses_gd_temp
                    eval_acc_gd += eval_acc_gd_temp
                    eval_top_3_freq_gd += eval_top_3_freq_gd_temp
                    if config.data_kernel != 'imagenet':
                        eval_prob_dist_gd += eval_prob_dist_gd_temp

                eval_losses_gd /= num_minibatches
                eval_acc_gd /= num_minibatches
                eval_top_3_freq_gd /= num_minibatches
                if config.data_kernel != 'imagenet':
                    eval_prob_dist_gd /= num_minibatches
            else:
                val_losses_gd, _, _, val_acc_gd, val_top_3_freq_gd, val_prob_dist_gd = predict_test.apply(params_gd,
                                                                                                              val_rng,
                                                                                                              val_data,
                                                                                                              True)

                eval_losses_gd, _, _, eval_acc_gd, eval_top_3_freq_gd, eval_prob_dist_gd = predict_test.apply(params_gd,
                                                                                                              eval_rng,
                                                                                                              eval_data,
                                                                                                              True)
            val_gd_losses.append(val_losses_gd)
            val_gd_acc.append(val_acc_gd)
            val_gd_top_3_freq.append(val_top_3_freq_gd)
            val_gd_prob_dist.append(val_prob_dist_gd)
            val_mean_max_prob_list.append(val_mean_max_prob)
            val_mean_target_prob_list.append(val_mean_target_prob)

            eval_gd_losses.append(eval_losses_gd)
            eval_gd_acc.append(eval_acc_gd)
            eval_gd_top_3_freq.append(eval_top_3_freq_gd)
            eval_gd_prob_dist.append(eval_prob_dist_gd)
            eval_mean_max_prob_list.append(eval_mean_max_prob)
            eval_mean_target_prob_list.append(eval_mean_target_prob)

            if config.data_kernel == 'imagenet':
                display(("Training step", step,
                         "Val Loss of GD Constructed (we learn alpha)", val_losses_gd.item(),
                         "Val GD Accuracy", val_acc_gd.item(),
                         "Val GD Top 3 Frequency", val_top_3_freq_gd.item(),
                         "Eval Loss of GD Constructed (we learn alpha)", eval_losses_gd.item(),
                         "Eval GD Accuracy", eval_acc_gd.item(),
                         "Eval GD Top 3 Frequency", eval_top_3_freq_gd.item()),
                        display_id="Learned GD")
            else:
                display(("Training step", step,
                         "Val Loss of GD Constructed (we learn alpha)", val_losses_gd.item(),
                         "Val GD Accuracy", val_acc_gd.item(),
                         "Val GD Top 3 Frequency", val_top_3_freq_gd.item(),
                         "Val GD Probability Loss", val_prob_dist_gd.item(),
                         "Val Mean Max Prob", val_mean_max_prob.item(),
                         "Val Mean Target Prob", val_mean_target_prob.item(),
                         "Eval Loss of GD Constructed (we learn alpha)", eval_losses_gd.item(),
                         "Eval GD Accuracy", eval_acc_gd.item(),
                         "Eval GD Top 3 Frequency", eval_top_3_freq_gd.item(),
                         "Eval GD Probability Loss", eval_prob_dist_gd.item(),
                         "Eval Mean Max Prob", eval_mean_max_prob.item(),
                         "Eval Mean Target Prob", eval_mean_target_prob.item()),
                        display_id="Learned GD")

            if val_losses_gd < best_val_loss:
                best_params_gd = params_gd.copy()
                best_step = step
                best_val_loss = val_losses_gd
                best_val_acc = val_acc_gd
                best_val_mse = val_prob_dist_gd
                best_val_top_3_freq = val_top_3_freq_gd

                waiting_time = 0
            else:
                waiting_time += 1

        # if config.early_stopping and waiting_time > patience:
        #     break

    if not config.early_stopping:
        best_params_gd = params_gd
        best_step = train_steps - 1

    return best_params_gd, train_data_rng, val_gd_losses, val_gd_acc, val_gd_top_3_freq, val_gd_prob_dist, \
        val_mean_max_prob_list, val_mean_target_prob_list, eval_gd_losses, eval_gd_acc, eval_gd_top_3_freq, \
        eval_gd_prob_dist, eval_mean_max_prob_list, eval_mean_target_prob_list, best_step


def train(_):
    """Train loop."""
    print("Use notebook to run the code")


if __name__ == '__main__':
    app.run()
