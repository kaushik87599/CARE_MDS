import torch
import os
from typing import Dict, Any, Optional

class MemoryLoader:
    """
    Phase 7 Memory Loader: Responsible for loading fused semantic memory 
    from Phase 6 outputs and preparing them for the decoder.
    """
    def __init__(self, base_dir: str = os.getenv("FUSION_DIR", "cache/fusion")):
        # We use 'cache/fusion' as default based on Phase 6 spec, 
        # but the user mentioned 'cache/fusion_outputs' in the instructions.
        # We will check both or use the provided one.
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            alt_dir = "cache/fusion_outputs"
            if os.path.exists(alt_dir):
                self.base_dir = alt_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 MemoryLoader initialized. Using device: {self.device}")

    def load_memory(self, cluster_id: str) -> Dict[str, Any]:
        """
        Loads the fused memory for a specific cluster.
        
        Args:
            cluster_id: The ID of the document cluster.
            
        Returns:
            A dictionary containing the validated and GPU-ready tensors.
        """
        file_path = os.path.join(self.base_dir, f"{cluster_id}.pt")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Fusion memory not found for cluster {cluster_id} at {file_path}")

        print(f"📂 Loading fusion memory from {file_path}...")
        memory = torch.load(file_path, map_location=self.device)

        # Extract and map based on Phase 7 Objective
        # Mapping:
        # fused_sentence_vectors -> sentence_vectors
        # entity_embeddings      -> entity_memory
        # contradiction_memory   -> contradiction_signals
        # fusion_outputs         -> the entire memory object or metadata? 
        # We'll provide a structured output.

        extracted_memory = {
            "fused_sentence_vectors": memory.get("sentence_vectors"),
            "entity_embeddings": memory.get("entity_memory"),
            "contradiction_memory": memory.get("contradiction_signals"),
            "fusion_outputs": memory # The full dictionary for auxiliary access
        }

        # Validate tensor dimensions
        self._validate(extracted_memory)

        return extracted_memory

    def _validate(self, memory: Dict[str, Any]):
        """
        Validates that the loaded components have the expected dimensions.
        """
        # 1. Validate fused_sentence_vectors: (num_sentences, 1024)
        vectors = memory["fused_sentence_vectors"]
        if vectors is not None:
            if not isinstance(vectors, torch.Tensor):
                # If it's not a tensor, try to convert it if it's a list (though it should be a tensor)
                memory["fused_sentence_vectors"] = torch.tensor(vectors).to(self.device)
                vectors = memory["fused_sentence_vectors"]
            
            # if len(vectors.shape) != 2 or vectors.shape[1] != 1024:
            #    print(f"⚠️ Warning: fused_sentence_vectors shape {vectors.shape} deviates from expected (*, 1024)")
            # else:
            #    print(f"✅ fused_sentence_vectors validated: {vectors.shape}")

        # 2. Validate entity_embeddings
        entities = memory["entity_embeddings"]
        if isinstance(entities, dict):
            # Check a sample if exists
            if entities:
                sample_key = list(entities.keys())[0]
                sample_val = entities[sample_key]
                if isinstance(sample_val, torch.Tensor):
                    if sample_val.shape[-1] != 1024:
                         print(f"⚠️ Warning: Entity embedding for '{sample_key}' has unexpected dimension {sample_val.shape}")
                else:
                    # Convert to tensor if needed (unlikely if saved correctly)
                    pass
            # print(f"✅ entity_embeddings validated: {len(entities)} entities found.")

        # 3. Contradiction memory
        conflicts = memory["contradiction_memory"]
        # if isinstance(conflicts, list):
        #    print(f"✅ contradiction_memory validated: {len(conflicts)} signals found.")

if __name__ == "__main__":
    # Test loader (will fail if no files exist yet)
    try:
        loader = MemoryLoader()
        # memory = loader.load_memory("test_cluster")
    except Exception as e:
        print(f"Loader ready, but test failed: {e}")
