from ml_collections import config_dict

config = config_dict.ConfigDict()

config.model_type = 'interleaved'
config.seed = 1
config.log_every_step = 100
config.create_plots = False
config.tensorboard_dir = './local/'
config.local_usage = False
config.analyse = True
config.ana_copy = False

# Language hps
config.vocab_size = 60000
config.vocab_dim = 128
config.warm_up = "False"
config.initial_lr = 0.00025
config.peak_value = 0.001
config.decay_rate = 0.8
config.warmup_steps = 4000
config.end_value = 0.00001
config.b1 = 0.9
config.b2 = 0.98
config.adam_eps = 1e-09
config.flip = False

# Data hps
config.dataset_size = 10
config.input_size = 10
config.e_size = 5
config.test_size = 100
config.input_range = 1.0
config.weight_scale = 1
config.input_shift = 0
config.size_distract = 0
config.classic_token_const = False
config.non_linear_reg_task = False

# kernel
config.data_kernel = 'linear'
config.model_kernel = 'linear'
config.j = 5
config.lam = 1e3
config.gamma = 5.0
config.sigma = 1.0

# Model hps
config.new_token_construction = True
config.deq = True

config.att_only_trans = True
config.include_query = False
config.use_bias = False
config.out_proj = False
config.in_proj = False
config.layer_norm = False
config.num_layers = 1
config.num_heads_gd = 1
config.num_heads_tr = 1
config.key_size = 11
config.output_size = 1
config.init_scale = 0.002 / config.num_layers
config.dampening = 1.0
config.gd_dampening = 1.0
config.clip = 0.0

config.bias_head = False
config.bias_data = False

config.gd_deq = True
config.sum_norm = False

config.one_hot_emb = False

config.mlp_output_dim = 5
config.second_layer = False
config.widening_factor = 10

# Training hps
config.adam = True
config.pre_train_gd = False
config.train_gd_whitening = True
config.train_gd_lr = True
config.lr = 0.001
config.dropout_rate = 0.0
config.bs = 512
config.bs_gd_train = 2048
config.grad_clip_value = 1e10
config.grad_clip_value_gd = 1e10
config.training_steps = 10000
config.training_steps_gd_constructed = 10000
config.wd = 0.0
config.cycle_data = 0

config.transition_steps = 500
config.gd_lr_decay_rate = 0.8
config.lr_decay_rate = 0.9
config.transition_begin = 0
config.staircase = True
config.use_lr_schedule = True

# Data loading
config.windowing = False
