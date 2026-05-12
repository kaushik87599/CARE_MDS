import os
import torch
import re
from tqdm import tqdm

INPUT_DIR = "cache/encoder_outputs"
OUTPUT_DIR = "cache/encoder_shards"
SHARD_SIZE = 100

def get_next_shard_idx():
    """Finds the highest existing shard index to prevent overwriting."""
    if not os.path.exists(OUTPUT_DIR):
        return 0
    shards = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("shard_") and f.endswith(".pt")]
    if not shards:
        return 0
    # Extract numbers from filenames like shard_015.pt
    indices = [int(re.search(r'\d+', f).group()) for f in shards]
    return max(indices) + 1

def save_and_clean(idx, data, paths):
    shard_path = os.path.join(OUTPUT_DIR, f"shard_{idx:03d}.pt")
    torch.save(data, shard_path)
    
    if os.path.exists(shard_path):
        for p in paths:
            try:
                os.remove(p)
            except OSError as e:
                print(f"Warning: Could not delete {p}: {e}")
    else:
        raise IOError(f"Critical Error: Failed to verify saved shard at {shard_path}.")

def run_sharding():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found.")
        return

    # Sort files numerically: 0.pt, 1.pt ... 3999.pt
    files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.split(".")[0])
    )

    if not files:
        print("No .pt files found to shard. Process may be complete.")
        return

    # RESUME LOGIC: Start from the next available index
    shard_idx = get_next_shard_idx()
    shard_data = []
    current_batch_files = []

    print(f"Resuming at Shard {shard_idx}. Found {len(files)} files remaining.")

    for file_name in tqdm(files, desc="Sharding Progress"):
        file_path = os.path.join(INPUT_DIR, file_name)
        try:
            # Using weights_only=True is safer/faster if possible
            data = torch.load(file_path, weights_only=False)
            shard_data.append(data)
            current_batch_files.append(file_path)

            if len(shard_data) == SHARD_SIZE:
                save_and_clean(shard_idx, shard_data, current_batch_files)
                shard_data = []
                current_batch_files = []
                shard_idx += 1
                
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            continue

    if shard_data:
        save_and_clean(shard_idx, shard_data, current_batch_files)

    print(f"\nSharding complete. Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_sharding()