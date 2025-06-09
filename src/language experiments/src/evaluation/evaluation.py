import os
from tqdm import tqdm
import torch
from src.util import get_device

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/checkpoints')

def generate_ending(model, tokenizer, sequence, use_beam=False, use_ngram_skip=False):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
        
    sequence = [token for token in sequence if token != tokenizer.pad_token_id and token != tokenizer.eos_token_id]
    sequence = torch.tensor(sequence).unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        input = sequence.to(model.device)
    
        if use_beam:
            output = model.beam_search(input, max_new_tokens=100, num_beams=3, eos_token_id=model.eos_token_id, ngram_skip_size=3 if use_ngram_skip else None)
        else:
            output = model.generate(input, max_new_tokens=100, eos_token_id=model.eos_token_id)
        
        return output[0].tolist()
    
def evaluate_models(models, tokenizer, dataloaders, num_generations=20, device=None, use_beam=True, use_ngram_skip=False):
    
    # Setup
    if device is None:
        device = get_device()
    
    for model in models:
        model.eos_token_id = tokenizer.eos_token_id
        model.device = device
        
    test_dataset = dataloaders['test']
    
    num_sequence = 0
    num_failed = 0
    
    for batch in test_dataset:
        for i in range(batch['input_ids'].size(0)):
            sequence = batch['input_ids'][i, :].tolist()
            
            if tokenizer.pad_token_id in sequence:
                sequence = sequence[:sequence.index(tokenizer.pad_token_id)]
            
            sequence_length = len(sequence)
            if sequence_length < 2:
                num_failed += 1
                continue
            
            true_beginning_ids = sequence[:sequence_length // 2]
            true_ending_ids = sequence[sequence_length // 2:]
            
            true_beginning_text = tokenizer.decode(true_beginning_ids)
            true_ending_text = tokenizer.decode(true_ending_ids)
            
            print('*' * 80)
            print(f'[True Story]\n{true_beginning_text} [{true_ending_text}]')
            print('=' * 80)
            
            for model in models:
                model.eval()
                model.to(device)
                generated_ending_ids = generate_ending(model, tokenizer, true_beginning_ids, use_beam=use_beam, use_ngram_skip=use_ngram_skip)
                generated_ending_text = tokenizer.decode(generated_ending_ids)
                print(f'[{model.config.get_name()}]\n{true_beginning_text} [{generated_ending_text}]')
                print('=' * 80)
            
            num_sequence += 1
            if num_sequence >= num_generations:
                break
        
        if num_sequence >= num_generations:
                break
    
    print(f'Generated {num_sequence} sequences ({num_failed} failed)')