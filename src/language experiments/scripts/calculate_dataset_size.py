import setup_paths
import time
from src.datasets import get_dataloaders, get_tokenizer
from src.util import get_time_remaining_formatted
tokenizer = get_tokenizer()
dataloaders = get_dataloaders(d_seq=512, tokenizer=tokenizer, batch_size=64)

train_dataloader = dataloaders['train']

total_tokens_including_padding = 0
total_tokens_excluding_padding = 0

num_batches = len(train_dataloader)
start_time = time.time()

for i, batch in enumerate(train_dataloader):
    tokens = batch['input_ids']
    total_tokens_including_padding += tokens.numel()
    total_tokens_excluding_padding += tokens[tokens != tokenizer.pad_token_id].numel()
    
    if i % 10 == 0:
        time_remaining = get_time_remaining_formatted(start_time, i, num_batches)
        print(f'\rProcessed {i}/{num_batches} batches. Time remaining: {time_remaining}', end='')
    
print(f'Total tokens including padding: {total_tokens_including_padding}')
print(f'Total tokens excluding padding: {total_tokens_excluding_padding}')