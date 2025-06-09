import torch

def get_device():
  
  device = None
  
  if torch.cuda.is_available():
    
    num_gpu = torch.cuda.device_count()
    print(f'Found {num_gpu} GPUs')
    
    for i in range(num_gpu):
    
      gpu = torch.device(f'cuda:{i}')
      free_memory, total_memory = torch.cuda.mem_get_info(gpu)
      total_memory = int(total_memory / 1024**3)
      free_memory = int(free_memory / 1024**3)  
      percent_used = (total_memory - free_memory) / total_memory
      
      print(f'[GPU {i}] Total memory: {total_memory}GB, Free memory: {free_memory}GB')
      
      if percent_used < 0.1:
        print(f"Using GPU {i}")
        return torch.device(f'cuda:{i}')
    
    if device is None:
      print("All GPUs are being used. Using CPU instead.")
      return torch.device('cpu')
      
    print("No GPUs found. Using CPU.")
    return torch.device('cpu')