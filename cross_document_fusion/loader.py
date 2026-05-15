import torch
import os
import pickle
from typing import Dict, Any, List, Generator

class DataLoader:
    """
    DataLoader for Phase 6: Cross-Document Fusion.
    Loads encoder hidden states, attention masks, and input IDs.
    Prioritizes shards for efficient batch processing.
    """
    def __init__(self, shard_dir: str = "cache/encoder_shards", output_dir: str = "cache/encoder_outputs"):
        self.shard_dir = shard_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure directories exist (at least one should eventually)
        if not os.path.exists(self.shard_dir) and not os.path.exists(self.output_dir):
            print(f"Warning: Neither {self.shard_dir} nor {self.output_dir} exist yet.")

    def load_shard(self, shard_file: str) -> List[Dict[str, Any]]:
        """Loads a shard file containing multiple clusters."""
        file_path = os.path.join(self.shard_dir, shard_file)
        try:
            # Crucial: Load to CPU to avoid filling GPU RAM with many clusters at once
            return torch.load(file_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading shard {file_path}: {e}")

    def load_individual_file(self, file_name: str) -> Dict[str, Any]:
        """Loads an individual cluster file."""
        file_path = os.path.join(self.output_dir, file_name)
        try:
            # Crucial: Load to CPU
            return torch.load(file_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")

    def stream_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        A generator that yields individual cluster data.
        Prioritizes shards, then falls back to individual files.
        """
        # 1. Process Shards (Primary)
        if os.path.exists(self.shard_dir):
            shard_files = sorted([f for f in os.listdir(self.shard_dir) if f.endswith(".pt")])
            for sf in shard_files:
                clusters = self.load_shard(sf)
                for cluster_data in clusters:
                    yield cluster_data

        # 2. Process Individual Outputs (Fallback)
        if os.path.exists(self.output_dir):
            individual_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith(".pt")])
            for f in individual_files:
                # Basic check: if we already processed shards, we might want to skip these
                # but for simplicity, we yield everything found in fallback if it exists.
                yield self.load_individual_file(f)

    def get_device(self):
        return self.device

    def load_packed_contexts(self, file_path: str = "cache/cache/packed_contexts.pkl") -> List[Dict[str, Any]]:
        """Loads the packed contexts from the pickle file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Packed contexts file not found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Quick verification logic
    loader = DataLoader()
    print("DataLoader initialized. Ready to load from cache/encoder_outputs/")
