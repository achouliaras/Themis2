import os, glob
import numpy as np
from typing import Any
from torch import Tensor


class StatisticsLogger:

    def __init__(self, mode: str = ""):
        self.mode = mode
        self.data = dict()

    def _add(self, key: str, value: Any):
        if value is None:
            return
        if isinstance(value, Tensor):
            value = value.item()
        if key not in self.data:
            self.data[key] = list()
        self.data[key].append(value)

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self._add(key, value)

    def to_dict(self):
        log_dict = dict()
        for key, value in self.data.items():
            log_dict.update({
                f"{self.mode}/{key}_mean": np.mean(value)
            })
        return log_dict


class LocalLogger(object):

    def __init__(self, path: str):
        self.path = path
        self.created = set()
        os.makedirs(self.path, exist_ok=True)

    def flush_all(self, prefix: str = None):
        """Explicitly erases only rollout.csv and train.csv in the logger's directory."""
        prefix = prefix + '_' if prefix else ''
        files_to_flush = [f'{prefix}rollout.csv', f'{prefix}train.csv']
        
        for filename in files_to_flush:
            file_path = os.path.join(self.path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                
        # Reset the in-memory set so headers get rewritten if these files are recreated
        self.created.clear()
        print(f"Flushed target logs in {self.path}")
    
    def truncate(self, target_value: float, prefix: str = None):
        """Truncates rollout.csv and train.csv in the logger's directory based on the time/total_timesteps column."""
        prefix = prefix + '_' if prefix else ''
        files_to_truncate = [f'{prefix}rollout.csv', f'{prefix}train.csv']
        
        for filename in files_to_truncate:
            log_path = os.path.join(self.path, filename)
            if not os.path.exists(log_path): 
                continue
                
            with open(log_path, 'r') as f:
                lines = f.readlines()
                
            # Skip if file is empty or missing our target column
            if not lines or "time/total_timesteps" not in lines[0]: 
                continue
                
            col_idx = lines[0].strip().split(',').index("time/total_timesteps")
            
            with open(log_path, 'w') as f:
                for i, line in enumerate(lines):
                    f.write(line)
                    # Stop writing if we hit or pass the target timestep
                    if i > 0 and float(line.strip().split(',')[col_idx]) >= target_value:
                        print(f"Truncated {filename} at timestep {target_value}")
                        break


    def write(self, log_data: dict, log_type: str):
        log_path = os.path.join(self.path, log_type + '.csv')
        if log_type not in self.created:
            if not os.path.exists(log_path):
                with open(log_path, 'w') as log_file:
                    log_file.write(','.join(log_data.keys()) + '\n')
            self.created.add(log_type)

        with open(log_path, 'a') as log_file:
            log_file.write(','.join(str(v) for v in log_data.values()) + '\n')
