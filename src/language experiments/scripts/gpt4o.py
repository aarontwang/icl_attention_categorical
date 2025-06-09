import setup_paths
import torch
from script_util import get_model_from_args, load_checkpoint, get_flags_from_args, get_models_from_name
from src.datasets import get_dataloaders, get_tokenizer
from src.evaluation import generate_gpt4o_inputs, create_batch, check_batch, cancel_batch, parse_batch

if __name__ == "__main__":
  
    torch.manual_seed(0)
    
    flags = get_flags_from_args()
    
    if 'input' in flags:
        print("Creating inputs")
        tokenizer = get_tokenizer()
        
        models = get_models_from_name()
        
        for model in models:
            model.resize_vocabulary(len(tokenizer))
            checkpoint = load_checkpoint(model)
            assert checkpoint is not None, f"No checkpoint found for model [{model.config.get_name()}]"
            model.load_state_dict(checkpoint['model_state_dicts'][checkpoint['epoch']])
            print(f'Loaded model [{model.config.get_name()}] with [{checkpoint["epoch"]}] epochs')
        
        dataloaders = get_dataloaders(d_seq=model.config.d_seq, tokenizer=tokenizer, batch_size=64)
        
        generate_gpt4o_inputs(models, tokenizer, dataloaders)
            
    elif 'batch' in flags:
        print("Creating batch")
        create_batch()
    elif 'check' in flags:
        check_batch()
    elif 'parse' in flags:
        parse_batch()
    else:
        raise ValueError("No valid flags detected. Please use --input or --batch")