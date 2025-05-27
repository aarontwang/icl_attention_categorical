"""Training fleixble Transformer model.

"""
import os
import sys
from functools import partial
from typing import Any, MutableMapping, NamedTuple, Tuple

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

import numpy as np
import haiku as hk
import jax
from jax import jit, vmap
import jax.numpy as jnp
import optax

from IPython.display import display

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from linear_nn import LinearNN
from data import create_img_data_linear, create_weights, create_w_e
from config import config

data_creator = vmap(create_img_data_linear, in_axes=(0, None, None, None, None, None, None), out_axes=0)


class TrainState(NamedTuple):
    """Container for the training state."""
    params: hk.Params
    opt_state: optax.OptState
    rng: jax.Array
    step: jax.Array


class TestState(NamedTuple):
    """Container for the test state."""
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

    if config.data_kernel == 'imagenet':
        data_creator = vmap(create_img_data_linear, in_axes=(0, None, None, None, None, None, None), out_axes=0)


def forward(tokens: jnp.ndarray):
    """Linear NN forward pass."""
    linear = LinearNN(output_dim=config.cats-1,
                      use_bias_p=False,
                      init_scale=config.init_scale)

    return linear(tokens)


def compute_loss(preds, targets):
    """Computes the negative log-likelihood (NLL) loss between predictions and targets."""
    # print(preds.shape)
    # print(targets.shape)

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
def loss_fn(data: jnp.ndarray) -> jnp.ndarray:
    """Computes the NLL loss between targets and predictions."""
    preds = forward(data[0])
    y_targets = data[-1]

    return compute_loss(preds, y_targets)


@hk.transform
def predict(data: jnp.ndarray) -> Tuple[jnp.ndarray]:
    """Predict."""
    preds = forward(data)

    return preds


@hk.transform
def predict_test(data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Predict test data used for analyses as well as metrics computation."""
    preds = forward(data[0])

    y_targets = data[-1]

    probs = preds

    loss_final = compute_loss(probs, y_targets)

    # accuracy
    preds_cat = jnp.argmax(probs, axis=-1).squeeze()
    targets_cat = jnp.argmax(y_targets, axis=-1)

    # print(targets_cat.shape)

    acc = jnp.where(preds_cat == targets_cat, 1, 0).sum(axis=None) / targets_cat.shape[0]

    # top-3 frequency
    top_3_preds = jnp.argsort(probs, axis=-1)[:, -3:]
    top_3_freq = jnp.where(targets_cat[:, None] == top_3_preds, 1, 0).sum(axis=None) / targets_cat.shape[0]

    return loss_final, acc, top_3_freq


@partial(jax.jit, static_argnums=(2))
def update(state: TrainState, data, optimizer) -> Tuple[TrainState, _Metrics]:
    """Does an SGD step and returns training state as well as metrics."""
    rng, new_rng = jax.random.split(state.rng)
    jit_loss_apply = jit(loss_fn.apply, static_argnums=1)

    loss_and_grad_fn = jax.value_and_grad(jit_loss_apply)
    loss, gradients = loss_and_grad_fn(state.params, rng, data)

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
               test_state: TestState, data) -> TestState:
    """Compute predictions from model."""
    rng, new_rng = jax.random.split(test_state.rng)
    loss, _, _ = predict_test.apply(train_state.params, rng, data)
    new_state = TestState(
        test_loss=loss,
        rng=new_rng,
        step=test_state.step + 1,
    )

    return new_state


def init_model(rng, train_data, test_data, optimizer) -> TrainState:
    """Initialize haiku transform modules to create train and test state."""
    train_rng, test_rng = jax.random.split(rng, num=2)
    initial_params = loss_fn.init(rng, train_data)

    initial_test_loss, _, _ = predict_test.apply(
        initial_params,
        rng,
        test_data)
    _ = predict.apply(initial_params, rng, test_data[0])

    initial_opt_state = optimizer.init(initial_params)

    return TrainState(
        params=initial_params,
        opt_state=initial_opt_state,
        rng=train_rng,
        step=np.array(0)), TestState(
        test_loss=initial_test_loss,
        rng=test_rng,
        step=np.array(0))


def init():
    """Initialize data creator, model, optimizer, etc."""
    rng = jax.random.PRNGKey(config.seed)
    rng, train_rng = jax.random.split(rng, 2)

    if config.data_kernel == 'imagenet':
        sharding = PositionalSharding(mesh_utils.create_device_mesh((2, 1)))

        train_data = data_creator(jax.device_put(jax.random.split(train_rng, num=config.bs), sharding.reshape(2, 1)),
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

    # print(train_data[0].shape)
    # print(train_data[-1].shape)

    train_data = tuple(data[0, :-1] for data in train_data)

    # print("train x shape: ", train_data[0].shape)
    # print("train y shape: ", train_data[-1].shape)

    train_state, test_state = init_model(rng, train_data, train_data, optimizer)

    return optimizer, train_state, test_state, rng
