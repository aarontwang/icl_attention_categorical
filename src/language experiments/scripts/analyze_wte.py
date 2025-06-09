import setup_paths
import torch
import torch.nn.functional as F
from script_util import get_model_from_args, load_checkpoint, get_flags_from_args
from src.datasets import get_dataloaders, get_tokenizer
from src.evaluation import evaluate_model

if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    model = get_model_from_args()
    checkpoint = load_checkpoint(model)
    
    assert checkpoint is not None, "No checkpoint found"
    
    device = None
    flags = get_flags_from_args()
    if 'cpu' in flags:
        device = torch.device('cpu')
    
    model.load_state_dict(checkpoint['model_state_dicts'][-1])
    
    wte = model.wte.weight
    
    norm_of_mean = wte.mean(dim=0).norm()
    mean_norm = wte.norm(dim=1).mean()
    relative_norm = norm_of_mean / mean_norm
    
    print(f'Norm of mean: {norm_of_mean}')
    print(f'Mean norm: {mean_norm}')
    print(f'Relative norm: {relative_norm}')