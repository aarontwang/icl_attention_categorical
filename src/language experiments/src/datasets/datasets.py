import os
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset, concatenate_datasets, load_from_disk
from transformers import GPT2TokenizerFast

DATASET_PATH = os.path.join(os.path.dirname(__file__), '../../data/datasets')

def build_dataset_splits(dataset, val_size=10000, test_size=10000):
    if isinstance(dataset, DatasetDict):
        dataset = concatenate_datasets([dataset[split] for split in dataset.keys()])
    
    train_val_split = dataset.train_test_split(test_size=val_size, shuffle=True)
    train_test_split = train_val_split['train'].train_test_split(test_size=test_size, shuffle=True)
    
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'valid': train_test_split['test'],
        'test': train_val_split['test']
    })
    
    return dataset

def merge_eot_entries(example):
    new_text = []
    buffer = []
    
    for text in example['text']:
        buffer.append(text)
        if '<|endoftext|>' in text:
            new_text.append('\n'.join(buffer))
            buffer = []
    
    return {'text': new_text}

def add_eot_tokens(example):
    return {'text': [f'{text}\n<|endoftext|>' for text in example['text']]}
    
def generate_tiny_stories_dataset():
    file_path = f'{DATASET_PATH}/tiny_stories/raw'
    if os.path.exists(file_path):
        return load_from_disk(file_path)
    dataset = concatenate_datasets([
        load_dataset("text", data_files="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt")['train'],
        load_dataset("text", data_files="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt")['train']
    ])
    dataset = dataset.map(merge_eot_entries, batched=True, batch_size=1024)
    dataset = build_dataset_splits(dataset)
    dataset.save_to_disk(file_path)
    return dataset

def generate_children_stories_dataset():
    file_path = f'{DATASET_PATH}/children_stories/raw'
    if os.path.exists(file_path):
        return load_from_disk(file_path)
    dataset = load_dataset("ajibawa-2023/Children-Stories-Collection")
    dataset.remove_columns(['text_token_length', 'prompt'])
    dataset = dataset.map(add_eot_tokens, batched=True, batch_size=1024)
    dataset = build_dataset_splits(dataset)
    dataset.save_to_disk(file_path)
    return dataset

def generate_dataset_splits():
    
    datasets = [
        generate_tiny_stories_dataset(),
        generate_children_stories_dataset()
    ]
       
    combined_datasets = DatasetDict({
        'train': concatenate_datasets([dataset['train'] for dataset in datasets]).shuffle(),
        'valid': concatenate_datasets([dataset['valid'] for dataset in datasets]).shuffle(),
        'test': concatenate_datasets([dataset['test'] for dataset in datasets]).shuffle()
    })
    
    return combined_datasets

def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})        
    tokenizer.name = 'gpt2-tokenizer'
    print(f'Loaded tokenizer: {tokenizer.name} with {tokenizer.vocab_size} tokens')
    return tokenizer

def get_dataloaders(d_seq, tokenizer, batch_size):
    
    dataset = generate_dataset_splits()
    
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    def collate_fn(examples):
        texts = [example['text'] for example in examples]
        return tokenizer(texts, padding=True, truncation=True, max_length=d_seq, return_tensors='pt')
    
    return {
        'train': DataLoader(dataset['train'], batch_size=batch_size, collate_fn=collate_fn),
        'valid': DataLoader(dataset['valid'], batch_size=batch_size, collate_fn=collate_fn),
        'test': DataLoader(dataset['test'], batch_size=batch_size, collate_fn=collate_fn)
    }