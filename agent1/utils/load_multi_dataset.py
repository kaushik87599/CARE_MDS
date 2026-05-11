import sys
import os
from datasets import load_dataset
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from preprocessing import check_dataset

def load_multi_dataset()->pd.DataFrame:
    datasets = check_dataset('datasets/datasets')
    multi_news = datasets["Multi-News"]["train"]
    df = pd.DataFrame(multi_news)
    return df
