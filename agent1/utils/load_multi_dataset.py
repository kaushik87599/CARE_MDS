import sys
import os
from datasets import load_dataset
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from preprocessing import check_dataset

# Configuration for dataset sampling
SAMPLING_TARGETS = {
    'CNN/Daily-News': {'train': 8000, 'validation': 1000, 'test': 1000},
    'Multi-News': {'train': 4000, 'validation': 500, 'test': 500}
}

def load_sampled_dataset(dataset_name: str = 'Multi-News', split: str = 'train') -> pd.DataFrame:
    """
    Loads a dataset and applies downsampling according to SAMPLING_TARGETS.
    """
    datasets = check_dataset('datasets/')
    
    if dataset_name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
    
    if split not in datasets[dataset_name]:
        available_splits = list(datasets[dataset_name].keys())
        raise ValueError(f"Split '{split}' not found in {dataset_name}. Available splits: {available_splits}")
        
    ds = datasets[dataset_name][split]
    
    # Apply Downsampling
    target_size = SAMPLING_TARGETS.get(dataset_name, {}).get(split)
    if target_size and len(ds) > target_size:
        print(f"Sampling {target_size} records from {dataset_name} ({split})...")
        ds = ds.select(range(target_size))
        
    df = pd.DataFrame(ds)
    return df

def load_multi_dataset() -> pd.DataFrame:
    """
    Legacy wrapper for Agent 1 runner.
    """
    return load_sampled_dataset('Multi-News', 'train')
