import time

def get_time_remaining_formatted(start_time, current_step, total_steps):
    elapsed_time = time.time() - start_time
    steps_remaining = total_steps - current_step
    if current_step == 0:
        current_step = 1
    time_remaining = elapsed_time / current_step * steps_remaining
    
    days = time_remaining // 86400
    hours = time_remaining // 3600 % 24
    minutes = time_remaining // 60 % 60
    seconds = time_remaining % 60
    
    time_str = f'{int(seconds)}s'
    if minutes > 0:
        time_str = f'{int(minutes)}m {time_str}'
    if hours > 0:
        time_str = f'{int(hours)}h {time_str}'
    if days > 0:
        time_str = f'{int(days)}d {time_str}'
    
    return time_str