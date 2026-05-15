import torch
import os
import pickle
from typing import Dict, Any, List, Generator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataLoader:
    """
    DataLoader for Phase 6: Cross-Document Fusion.
    Loads encoder hidden states, attention masks, and input IDs.
    """
    def __init__(self, shard_dir: str = None, output_dir: str = None):
        self.shard_dir = shard_dir or os.getenv("SHARD_DIR", "cache/encoder_shards")
        self.output_dir = output_dir or os.getenv("ENCODER_OUT_DIR", "cache/encoder_outputs")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_shard(self, shard_file: str) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.shard_dir, shard_file)
        try:
            return torch.load(file_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading shard {file_path}: {e}")

    def load_individual_file(self, file_name: str) -> Dict[str, Any]:
        file_path = os.path.join(self.output_dir, file_name)
        try:
            return torch.load(file_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")

    def stream_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        A generator that yields individual cluster data.
        Prioritizes shards, then falls back to individual files.
        """
        if os.path.exists(self.shard_dir):
            shard_files = sorted([f for f in os.listdir(self.shard_dir) if f.endswith(".pt")])
            for sf in shard_files:
                clusters = self.load_shard(sf)
                for cluster_data in clusters:
                    yield cluster_data

        if os.path.exists(self.output_dir):
            individual_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith(".pt")])
            for f in individual_files:
                yield self.load_individual_file(f)

    def get_device(self):
        return self.device

    def load_packed_contexts(self, file_path: str = None) -> List[Dict[str, Any]]:
        """Loads the packed contexts."""
        if file_path is None:
            packed_cache_dir = os.getenv("PACKED_CACHE_DIR", "cache/cache")
            file_path = os.path.join(packed_cache_dir, "packed_contexts.pkl")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Packed contexts file not found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    loader = DataLoader()
    print("DataLoader initialized.")
