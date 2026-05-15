import os
import torch
import re
import gc
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use centralized paths
INPUT_DIR = os.getenv("ENCODER_OUT_DIR", "cache/encoder_outputs")
OUTPUT_DIR = os.getenv("SHARD_DIR", "cache/encoder_shards")
SHARD_SIZE = 100

def get_next_shard_idx():
    """Finds the highest existing shard index to prevent overwriting."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return 0
    shards = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("shard_") and f.endswith(".pt")]
    if not shards:
        return 0
    indices = [int(re.search(r'\d+', f).group()) for f in shards]
    return max(indices) + 1

def save_and_clean(idx, data, paths):
    shard_path = os.path.join(OUTPUT_DIR, f"shard_{idx:03d}.pt")
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    
    print(f"\n💾 Saving Shard {idx} ({len(data)} items)...")
    torch.save(data, shard_path)
    
    if os.path.exists(shard_path):
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass 
        return True
    return False

def run_sharding():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found.")
        return

    files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split(".")[0])
    )

    if not files:
        print("No .pt files found to shard.")
        return

    existing_indices = set(int(f.split(".")[0]) for f in files)
    min_idx = min(existing_indices)
    max_idx = max(existing_indices)
    
    print(f"Found {len(files)} .pt files to shard (Indices {min_idx} to {max_idx}).")

    shard_idx = get_next_shard_idx()
    shard_data = []
    current_batch_files = []

    print(f"Resuming at Shard {shard_idx}. Found {len(files)} files remaining.")

    for file_name in tqdm(files, desc="Sharding Progress"):
        file_path = os.path.join(INPUT_DIR, file_name)
        try:
            data = torch.load(file_path, weights_only=False, map_location='cpu')
            shard_data.append(data)
            current_batch_files.append(file_path)

            if len(shard_data) >= SHARD_SIZE:
                save_and_clean(shard_idx, shard_data, current_batch_files)
                del shard_data
                del current_batch_files
                gc.collect()
                
                shard_data = []
                current_batch_files = []
                shard_idx += 1
                
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            continue

    if shard_data:
        save_and_clean(shard_idx, shard_data, current_batch_files)
        del shard_data
        gc.collect()

    print(f"\nSharding complete. Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_sharding()