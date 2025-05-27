import sys

import matplotlib.pylab as pl
import matplotlib.colors as mcolors

# import written code
from transformer import Transformer
from data import create_reg_data, create_reg_data_kernel, create_reg_data_classic_token, create_weights
from config import config
from train import *
from eval import *
from util import *

from IPython.display import display

def get_results_dict():
    results = {'loss_trans_list': [[y.tolist() for y in x] for x in loss_trans_list]}
    results['loss_trans_train_list'] = loss_trans_train_list
    results['losses_gd_list'] = [[y.tolist() for y in x] for x in losses_gd_list]
    results['losses_gd_list_trained'] = [[y.tolist() for y in x] for x in losses_gd_list_trained]
    results['losses_int_list_trained'] = [[y.tolist() for y in x] for x in losses_int_list_trained]
    results['cos_sim_list'] = [[y.tolist() for y in x] for x in cos_sim_list]
    results['cos_sim_list_o'] = [[y.tolist() for y in x] for x in cos_sim_list_o]
    results['grad_norm_list'] = [[y.tolist() for y in x] for x in grad_norm_list]
    results['grad_norm_list_o'] = [[y.tolist() for y in x] for x in grad_norm_list_o]
    results['p_norm_list'] = [[y.tolist() for y in x] for x in p_norm_list]
    results['p_norm_list_o'] = [[y.tolist() for y in x] for x in p_norm_list_o]

    results['sig_dist_list'] = [[y.tolist() for y in x] for x in sig_dist_list]
    results['sig_dist_list_trained'] = [[y.tolist() for y in x] for x in sig_dist_list_trained]
    results['sig_dist_trans_list'] = [[y.tolist() for y in x] for x in sig_dist_trans_list]

    results['ir_t_list'] = [[y.tolist() for y in x] for x in ir_t_list]
    results['ws_t_list'] = [[y.tolist() for y in x] for x in ws_t_list]
    results['ir_gd_list'] = [[y.tolist() for y in x] for x in ir_gd_list]
    results['ws_gd_list'] = [[y.tolist() for y in x] for x in ws_gd_list]

    results['ir_t_ood_list'] = [[y.tolist() for y in x] for x in ir_t_ood_list]
    results['ws_t_ood_list'] = [[y.tolist() for y in x] for x in ws_t_ood_list]
    results['ir_gd_ood_list'] = [[y.tolist() for y in x] for x in ir_gd_ood_list]
    results['ws_gd_ood_list'] = [[y.tolist() for y in x] for x in ws_gd_ood_list]

    results['ir_gd_trained_list'] = [[y.tolist() for y in x] for x in ir_gd_trained_list]
    results['ws_gd_trained_list'] = [[y.tolist() for y in x] for x in ws_gd_trained_list]
    results['ir_gd_ood_trained_list'] = [[y.tolist() for y in x] for x in ir_gd_ood_trained_list]
    results['ws_gd_ood_trained_list'] = [[y.tolist() for y in x] for x in ws_gd_ood_trained_list]

    results['ir_inter_list'] = [[y.tolist() for y in x] for x in ir_inter_list]
    results['ws_inter_list'] = [[y.tolist() for y in x] for x in ws_inter_list]
    results['ir_inter_ood_list'] = [[y.tolist() for y in x] for x in ir_inter_ood_list]
    results['ws_inter_ood_list'] = [[y.tolist() for y in x] for x in ws_inter_ood_list]

    results['losses_noisy_list'] = [[[z.tolist() for z in y] for y in x] for x in losses_noisy_list]
    results['losses_gd_noisy_list'] = [[[z.tolist() for z in y] for y in x] for x in losses_gd_noisy_list]
    results['losses_gd_noisy_trained_list'] = [[[z.tolist() for z in y] for y in x] for x in
                                               losses_gd_noisy_trained_list]
    results['losses_inter_noisy_list'] = [[[z.tolist() for z in y] for y in x] for x in losses_inter_noisy_list]

    return results


# @title Logic how to interpolate weights
def interpolate_weights(train_state, params_gd, deq=False):
    if (config.num_heads == 1 and
            config.sum_norm == False and config.deq == True and
            config.layer_norm == False and config.att_only_trans == True):

        cur_train_params = {k.replace('transformer', 'Transformer_gd'): v.copy() for
                            k, v in train_state.params.items()}

        inter_params = {k.replace('transformer', 'Transformer_gd'): {'w': jnp.zeros_like(v['w'])} for
                        k, v in train_state.params.items()}

        for k, v in cur_train_params.items():
            if "key" in k:
                key_gd = params_gd[k]['w'].copy()
                key = cur_train_params[k]['w'].copy()
            if "linear" in k:
                linear_gd = params_gd[k]['w'].copy()
                linear = cur_train_params[k]['w'].copy()
            if "query" in k:
                query_gd = params_gd[k]['w'].copy()
                query = cur_train_params[k]['w'].copy()
            if "value" in k:
                value_gd = params_gd[k]['w'].copy()
                value = cur_train_params[k]['w'].copy()

                query = jnp.matmul(query, key.T)
                # print(query)
                key = jnp.identity(query.shape[0])
                mean = np.mean([query[a, a] for a in range(query.shape[0] - 1)])
                query = query / mean
                query_gd = jnp.matmul(query_gd, key.T)
                key_gd = jnp.identity(query.shape[0])
                query = (query + query_gd) / 2

                linear = jnp.matmul(value, linear)
                # print(linear)
                value = jnp.identity(query.shape[0])
                linear = linear * mean
                linear_gd = jnp.matmul(value_gd, linear_gd)
                value_gd = jnp.identity(query.shape[0])
                linear = (linear + linear_gd) / 2

                inter_params[k.replace('value', 'linear')]['w'] = linear
                inter_params[k.replace('value', 'value')]['w'] = value
                inter_params[k.replace('value', 'query')]['w'] = query
                inter_params[k.replace('value', 'key')]['w'] = key

        losses_int, _, _, _ = predict_test.apply(inter_params, eval_rng, eval_data, True, config.use_kernel)
    else:
        losses_int = None
        inter_params = None
    return losses_int, inter_params


if __name__ == '__main__':
    colors = pl.colormaps['Dark2']

    pl.rcParams.update({'font.size': 12})
    pl.rc('axes', labelsize=14)
    pl.rcParams.update({
        "text.usetex": False,
    })

    conf_init(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4])), bool(int(sys.argv[5])))

    change_dataloader()

    if config.use_kernel:
        data_creator = vmap(create_reg_data_kernel,
                            in_axes=(0, None, None, None, None, None, None, None), out_axes=0)

    loss_trans_list = [[] for _ in range(config.num_seeds)]
    loss_trans_train_list = [[] for _ in range(config.num_seeds)]
    losses_gd_list = [[] for _ in range(config.num_seeds)]
    losses_gd_list_trained = [[] for _ in range(config.num_seeds)]
    losses_int_list_trained = [[] for _ in range(config.num_seeds)]
    cos_sim_list, cos_sim_list_o = [[] for _ in range(config.num_seeds)], [[] for _ in range(config.num_seeds)]
    grad_norm_list, grad_norm_list_o = [[] for _ in range(config.num_seeds)], [[] for _ in range(config.num_seeds)]
    p_norm_list, p_norm_list_o = [[] for _ in range(config.num_seeds)], [[] for _ in range(config.num_seeds)]

    sig_dist_list = [[] for _ in range(config.num_seeds)]
    sig_dist_list_trained = [[] for _ in range(config.num_seeds)]
    sig_dist_trans_list = [[] for _ in range(config.num_seeds)]

    cos_sim_list, cos_sim_list_o = [[] for _ in range(config.num_seeds)], [[] for _ in range(config.num_seeds)]
    grad_norm_list, grad_norm_list_o = [[] for _ in range(config.num_seeds)], [[] for _ in range(config.num_seeds)]
    p_norm_list, p_norm_list_o = [[] for _ in range(config.num_seeds)], [[] for _ in range(config.num_seeds)]

    ir_t_list = [[] for _ in range(config.num_seeds)]
    ws_t_list = [[] for _ in range(config.num_seeds)]
    ir_gd_list = [[] for _ in range(config.num_seeds)]
    ws_gd_list = [[] for _ in range(config.num_seeds)]

    ir_t_ood_list = [[] for _ in range(config.num_seeds)]
    ws_t_ood_list = [[] for _ in range(config.num_seeds)]
    ir_gd_ood_list = [[] for _ in range(config.num_seeds)]
    ws_gd_ood_list = [[] for _ in range(config.num_seeds)]

    ir_gd_trained_list = [[] for _ in range(config.num_seeds)]
    ws_gd_trained_list = [[] for _ in range(config.num_seeds)]
    ir_gd_ood_trained_list = [[] for _ in range(config.num_seeds)]
    ws_gd_ood_trained_list = [[] for _ in range(config.num_seeds)]

    ir_inter_list = [[] for _ in range(config.num_seeds)]
    ws_inter_list = [[] for _ in range(config.num_seeds)]
    ir_inter_ood_list = [[] for _ in range(config.num_seeds)]
    ws_inter_ood_list = [[] for _ in range(config.num_seeds)]

    losses_noisy_list = [[] for _ in range(config.num_seeds)]
    losses_gd_noisy_list = [[] for _ in range(config.num_seeds)]
    losses_gd_noisy_trained_list = [[] for _ in range(config.num_seeds)]
    losses_inter_noisy_list = [[] for _ in range(config.num_seeds)]

    # interpolate GD and trained TF
    inter = True if (config.deq and not config.use_softmax and config.num_heads == 1) else False

    eval_rng = jax.random.PRNGKey(5)
    for cur_seed in range(config.num_seeds):
        config.seed = cur_seed
        optimizer, train_state, _, rng = init()
        rng, data_rng = jax.random.split(rng, 2)
        if config.analyze:
            lr_min, min_loss = scan_lrs(eval_rng, lin_diag=False, bs=10000)
            if cur_seed == 0:
                print('Best lr found for ', config.num_layers, ' steps of gradient descent: ',
                      lr_min / config.dataset_size, " with loss ", min_loss)

            params_gd = create_weights(config.input_size, 1, config.dataset_size, lr_min,
                                       jax.random.normal(data_rng, shape=[1, 1, config.output_size]) * 0.0,
                                       lin_diag=False, gd_deq=config.gd_deq,
                                       num_layers=config.num_layers,
                                       input_mlp_rnd=rng if (config.input_mlp or config.in_proj) else None,
                                       in_proj=config.in_proj)
            if config.num_layers > 1 or (config.in_proj and config.num_layers == 1):
                if cur_seed == 0:
                    lr_min, min_loss = scan_lrs(eval_rng, lin_diag=True, bs=10000)
                    params_init = create_weights(config.input_size, 1, config.dataset_size, lr_min,
                                                 jax.random.normal(data_rng, shape=[1, 1, config.output_size]) * 0.0,
                                                 lin_diag=True, gd_deq=config.gd_deq,
                                                 num_layers=config.num_layers,
                                                 input_mlp_rnd=eval_rng if (
                                                             config.input_mlp or config.in_proj) else None,
                                                 in_proj=config.in_proj)
                    params_gd_trained, data_rng = pre_train_gd_hps(eval_rng, params_init)
            else:
                params_gd_trained = params_gd

        if config.use_kernel:
            eval_data = data_creator(jax.random.split(eval_rng, num=10000),
                                     config.input_size,
                                     config.dataset_size,
                                     config.size_distract,
                                     config.input_range,
                                     config.weight_scale,
                                     config.m,
                                     config.gamma)
        else:
            eval_data = data_creator(jax.random.split(eval_rng, num=10000),
                                     config.input_size,
                                     config.dataset_size,
                                     config.size_distract,
                                     config.input_range,
                                     config.weight_scale)

        if config.analyze:
            loss_gd, _, _, sig_dist = predict_test.apply(params_gd, eval_rng, eval_data, True, config.use_kernel)
            loss_gd_trained, _, _, sig_dist_trained = predict_test.apply(params_gd_trained, eval_rng,
                                                       eval_data, True, config.use_kernel)
        original_data_rng = data_rng
        for step in range(config.training_steps):
            if config.cycle_data > 0:
                if step % config.cycle_data == 0:
                    data_rng = original_data_rng

            rng, data_rng = jax.random.split(data_rng, 2)

            if config.use_kernel:
                train_data = data_creator(jax.random.split(rng, num=config.bs),
                                          config.input_size,
                                          config.dataset_size,
                                          config.size_distract,
                                          config.input_range,
                                          config.weight_scale,
                                          config.m,
                                          config.gamma)
            else:
                train_data = data_creator(jax.random.split(rng, num=config.bs),
                                          config.input_size,
                                          config.dataset_size,
                                          config.size_distract,
                                          config.input_range,
                                          config.weight_scale)

            train_state, metrics = update(train_state, train_data, optimizer)
            if step % 100 == 0:

                loss_trans, _, _, sig_dist_trans = predict_test.apply(train_state.params, eval_rng,
                                                      eval_data, False, config.use_kernel)
                loss_trans_list[cur_seed].append(loss_trans)
                loss_trans_train_list[cur_seed].append(metrics['train_loss'].item(), )

                sig_dist_trans_list[cur_seed].append(sig_dist_trans)

                if config.analyze:
                    losses_gd_list[cur_seed].append(loss_gd)
                    losses_gd_list_trained[cur_seed].append(loss_gd_trained)
                    sig_dist_list[cur_seed].append(sig_dist)
                    sig_dist_list_trained[cur_seed].append(sig_dist_trained)

                    losses_int, inter_params = interpolate_weights(train_state, params_gd_trained)

                    losses_int_list_trained[cur_seed].append(losses_int)

                    # rng, data_rng, eval_rng = jax.random.split(data_rng, 3)
                    # Alignment Transformers and GD
                    cos_sim, w_norm, p_norm = analyze(eval_data, train_state, eval_rng,
                                                      params_gd)
                    cos_sim_o, w_norm_o, p_norm_o = analyze(eval_data, train_state, eval_rng,
                                                            params_gd_trained)
                    if step > 0:
                        display(("Current seed", cur_seed,
                                 "Training step", step, "Gradient descent loss", loss_gd.item(),
                                 "GD ++ loss", loss_gd_trained.item(),
                                 "Trained TF loss", loss_trans.item(),
                                 "Interpolated model loss", losses_int.item() if inter else "-",
                                 "Cosine sim TF vs GD", cos_sim.item(),
                                 "Cosine sim TF vs GD++", cos_sim_o.item() if config.num_layers > 1 else "-",
                                 "GD sigmoid probability loss", sig_dist.item(),
                                 "Trained TF sigmoid probability loss", sig_dist_trans.item()),
                                display_id="Cur met")

                    cos_sim_list[cur_seed].append(cos_sim)
                    grad_norm_list[cur_seed].append(w_norm)
                    p_norm_list[cur_seed].append(p_norm)

                    cos_sim_list_o[cur_seed].append(cos_sim_o)
                    grad_norm_list_o[cur_seed].append(w_norm_o)
                    p_norm_list_o[cur_seed].append(p_norm_o)

                else:
                    print(step, loss_trans)

    if not os.path.isdir(config.save_folder):
        os.mkdir(config.save_folder)

    cosine_low = 0.0
    if config.num_layers == 1:
        # loss
        display_learning(loss_trans_list, test=[losses_gd_list[0]], y_lim_u=0.25, y_lim_l=0.0,
                         rw=1, title="train.jpg", folder=config.save_folder, allow_download=True,
                         single_seeds=True, label_title="Loss",
                         title2='GD', title1='Trained TF',
                         title3='GD', loc_first='upper right',
                         num_iter_os=len(loss_trans_list[0]) * 100)

        # sigmoid distance
        display_learning(sig_dist_trans_list, test=[sig_dist_list[0]], y_lim_u=0.25, y_lim_l=0.0,
                         rw=1, title="sig_loss.jpg", folder=config.save_folder, allow_download=True,
                         single_seeds=True, label_title="Sigmoid Loss",
                         title2='GD', title1='Trained TF (sigmoid)',
                         title3='GD', loc_first='upper right',
                         num_iter_os=len(loss_trans_list[0]) * 100)

        # similarity to GD
        display_learning(cos_sim_list, grad_norm_list, p_norm_list,
                         title1="Model cos",
                         title2="Model diff", y_lim_u=2,
                         title3="Preds diff", second_axis=True, color_add=0.2,
                         y_lim_u2=1.19, loc_sec='center right', single_seeds=False,
                         y_lim_l2=cosine_low, color_axis=False, width=5, y_label2='Cosine sim',
                         rw=1, num_iter_os=len(grad_norm_list[0]) * 100, title="sim.jpg",
                         folder=config.save_folder, allow_download=True)
    else:
        # loss
        display_learning(loss_trans_list, gt=losses_gd_list_trained,
                         test=[losses_gd_list[0]], y_lim_u=0.3, y_lim_l=0.0,
                         rw=1, title="train.jpg", folder=config.save_folder, allow_download=True,
                         title2='GD', title1='Trained TF',
                         title3='GD$^{++}$', loc_first='upper right', x_label="Training steps",
                         single_seeds=True, plot_title=None,
                         num_iter_os=len(loss_trans_list[0]) * 100)

        # sigmoid distance
        display_learning(sig_dist_trans_list, gt=sig_dist_list_trained,
                         test=[sig_dist_list[0]], y_lim_u=0.3, y_lim_l=0.0,
                         rw=1, title="sig_loss.jpg", folder=config.save_folder, allow_download=True,
                         title2='GD', title1='Trained TF (sigmoid)',
                         title3='GD$^{++}$', loc_first='upper right', x_label="Training steps",
                         single_seeds=True, plot_title=None,
                         num_iter_os=len(loss_trans_list[0]) * 100)

        # similarity to GD
        display_learning(cos_sim_list, grad_norm_list, p_norm_list, title1="Model cos",
                         title2="Model diff", y_lim_u=1.8,
                         title3="Preds diff", second_axis=True, color_add=0.2,
                         y_lim_u2=1.09999999, color_axis=False, width=4, x_label="Training steps",
                         plot_title="GD vs trained TF",
                         y_lim_l2=0.5, loc_sec='center right', y_label1='L2 Norm', y_label2='Cosine sim',
                         rw=1, num_iter_os=len(loss_trans_list[0]) * 100, title="sim_gd.jpg",
                         folder=config.save_folder, allow_download=True, plot_num=1)

        # similarity to GD++
        display_learning(cos_sim_list_o, grad_norm_list_o, p_norm_list_o, title1="Model cos",
                         title2="Model diff", y_lim_u=1.8, x_label="Training steps",
                         plot_title="GD$^{++}$ vs trained TF",
                         title3="Preds diff", second_axis=True, color_add=0.2,
                         y_lim_u2=1.0599999, color_axis=False, width=4, y_label1='L2 Norm', y_label2='Cosine sim',
                         y_lim_l2=0.5, loc_sec='center right',
                         rw=1, num_iter_os=len(loss_trans_list[0]) * 100, title="sim_gd_plus.jpg", folder=config.save_folder,
                         allow_download=True)

    try:
        results = get_results_dict()

        with open(config.save_folder + "/" + "results.json", "w") as fp:
            json.dump(results, fp)

        print("Done writing results into json file")
    except Exception as e:
        print("Error getting results")
        print("The error is: ", e)
