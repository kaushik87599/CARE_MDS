import os
import pickle
import numpy as np
import json
import torch
import pandas as pd
# cache set up

CACHE_DIR = 'cache'

# Helper to create directory structure
def setup_cache_dirs(dataset_names):
    for name in dataset_names:
        path = os.path.join(CACHE_DIR, name.lower().replace('-', '').replace('/', '_'))
        os.makedirs(path, exist_ok=True)
        print(f"Cache directory ready: {path}")

def save_cache(dataset_name, tokenized_data=None, entities=None, embeddings=None, analysis_results=None):
    # Clean name for filesystem compatibility
    # Specifically handle CNN and Multi-News to match user request
    if 'cnn' in dataset_name.lower():
        base = 'cnn'
    elif 'multi' in dataset_name.lower():
        base = 'multinews'
    else:
        base = dataset_name.lower().replace('-', '').replace('/', '_')
    
    # Handle split if present in name (e.g. "CNN/Daily-News_train")
    if '_' in dataset_name:
        split = dataset_name.split('_')[-1]
        base_path = os.path.join(CACHE_DIR, base, split)
    else:
        base_path = os.path.join(CACHE_DIR, base)
        
    os.makedirs(base_path, exist_ok=True)

    # 1. Store Tokenized Data (.pkl as requested)
    if tokenized_data is not None:
        with open(os.path.join(base_path, "tokenized.pkl"), "wb") as f:
            pickle.dump(tokenized_data, f)
        print(f"✅ Saved tokenized.pkl for {dataset_name}")

    # 2. Store Entity Extraction Results (.pkl as requested)
    if entities is not None:
        with open(os.path.join(base_path, "entities.pkl"), "wb") as f:
            pickle.dump(entities, f)
        print(f"✅ Saved entities.pkl for {dataset_name}")

    # 3. Store Sentence Embeddings (.npy as requested)
    if embeddings is not None:
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        np.save(os.path.join(base_path, "sentence_embeddings.npy"), np.array(embeddings))
        print(f"✅ Saved sentence_embeddings.npy for {dataset_name}")

    # 4. Store Intermediate Analysis (JSON)
    if analysis_results is not None:
        # If it's a dict of metrics, save it
        serializable_results = {}
        for k, v in analysis_results.items():
            if isinstance(v, (pd.Series, np.ndarray)):
                serializable_results[k] = v.tolist()
            elif torch.is_tensor(v):
                serializable_results[k] = v.cpu().tolist()
            else:
                serializable_results[k] = v

        with open(os.path.join(base_path, "analysis_results.json"), "w") as f:
            json.dump(serializable_results, f, indent=4)
        print(f"✅ Saved analysis_results.json for {dataset_name}")

# Initialize directories (if run as script)
if __name__ == "__main__":
    setup_cache_dirs(['cnn', 'multinews'])