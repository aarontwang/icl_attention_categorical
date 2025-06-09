import os
import sys
import importlib
import torch
from dataclasses import fields, is_dataclass

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/checkpoints')

def load_checkpoint(model):
    model_path = f'{CHECKPOINT_DIR}/{model.config.get_name()}_checkpoint.pth'
    if os.path.exists(model_path):
        return torch.load(model_path, weights_only=False)
    return None

def _get_attr_case_insensitive(module, name):
    name = name.replace('_', '')
    for attr in dir(module):
        if attr.lower() == name.lower():
            return getattr(module, attr)
    return None


def get_models_from_name():
    model_names = sys.argv[1:]
    models = []
    for model_name in model_names:
        if model_name.startswith('--'):
            continue
        model_name = model_name.replace("'", "").strip()
        model_params = model_name.split('_')
        model_type = model_params[0]
        d_seq = int(model_params[1][:-1])
        d_embed = int(model_params[2][:-1])
        n_head = int(model_params[3][:-1])
        n_layer = int(model_params[4][:-1])
        attn_fn = model_params[5]
        use_ff = model_params[6].lower() == 'ff=true'
        
        model_module = importlib.import_module('src.models')
        model_class = _get_attr_case_insensitive(model_module, model_type)
        model_config_class = _get_attr_case_insensitive(model_module, model_type + 'Config')
        
        assert model_class is not None
        assert model_config_class is not None
        
        config = model_config_class(d_seq=d_seq, d_embed=d_embed, n_head=n_head, n_layer=n_layer, attn_fn=attn_fn, use_ff=use_ff)
        
        models.append(model_class(config))
        
    return models
        
def get_model_from_args():
    
    args = sys.argv[1:]

    model_name = args[0]

    model_module = importlib.import_module('src.models')
    model_class = _get_attr_case_insensitive(model_module, model_name)
    model_config_class = _get_attr_case_insensitive(model_module, model_name + 'Config')
    
    assert model_class is not None, f"Model class {model_name} not found"
    assert model_config_class is not None, f"Model config class {model_name}Config not found"
    
    config = model_config_class()    
    
    for parameter in sys.argv[2:]:
        s = parameter.split('=')
        if len(s) != 2:
            continue
        key, value = s

        def get_all_fields(dataclass):
            # Collect all fields from a dataclass and its parent dataclasses
            if not is_dataclass(dataclass):
                return []
            fields_list = list(fields(dataclass))
            for base_class in dataclass.__bases__:
                fields_list.extend(get_all_fields(base_class))
            return fields_list

        # Check all fields in the hierarchy
        all_fields = {field.name for field in get_all_fields(model_config_class)}
        if key in all_fields:
            if type(getattr(config, key)) == int:
                value = int(value)
                setattr(config, key, value)
            elif type(getattr(config, key)) == bool:
                value = value.lower() == 'true'
                setattr(config, key, value)
            else:
                setattr(config, key, value)
    return model_class(config)

def get_flags_from_args():
    args = sys.argv[1:]
    flags = []
    for arg in args:
        if '=' in arg:
          continue
        if arg.startswith('--'):
            flags.append(arg[2:])
    return flags

import os

def setup_cache(cache_dir):
    print(f'Setting up local cache at {cache_dir}')
    os.environ['HF_HOME'] = f'{cache_dir}'
    os.environ['TRANSFORMERS_CACHE'] = f'{cache_dir}/transformers'
    os.environ['HF_DATASETS_CACHE'] = f'{cache_dir}/datasets'
    os.environ['TMPDIR'] = f'{cache_dir}'