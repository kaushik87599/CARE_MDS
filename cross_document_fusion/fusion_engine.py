import torch
import os
from typing import Dict, Any, List, Optional
from .loader import DataLoader
from .sentence_recovery import SentenceRepresenter
from .attention import CrossDocumentFusionLayer
from .entity_alignment import EntityAligner
from .contradiction_detection import ContradictionDetector
from .interaction import FusionInteraction

class FusionEngine:
    """
    Phase 6 Orchestrator: Fusion Engine.
    Combines all steps to build a unified Cross-Document Fusion Memory.
    """
    def __init__(self, tokenizer_path: str = "models/models/final_mds_led"):
        self.loader = DataLoader()
        self.device = self.loader.get_device()
        
        print(f"⚙️ Initializing Fusion Engine on {self.device}...")
        
        self.representer = SentenceRepresenter(tokenizer_path)
        
        # Optimization: Use Half precision for all layers if on GPU
        self.model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        self.fusion_layer = CrossDocumentFusionLayer().to(self.device).to(self.model_dtype)
        self.entity_aligner = EntityAligner()
        self.contradiction_detector = ContradictionDetector() # Will handle its own dtype/device
        self.interaction_layer = FusionInteraction().to(self.device).to(self.model_dtype)

    def fuse_cluster(self, cluster_data: Dict[str, Any], packed_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single cluster through the entire Phase 6 pipeline.
        
        Args:
            cluster_data: Dict containing hidden_states, input_ids from Phase 5.
            packed_context: Dict containing original sentences and metadata from Phase 4.
            
        Returns:
            fusion_memory: The reasoning substrate for Phase 7.
        """
        with torch.no_grad():
            # STEP 2 & 4: Recover Sentence-Level Representations and Document IDs
            sentence_vectors, doc_ids = self.representer.recover_vectors(
                cluster_data["hidden_states"], 
                cluster_data["input_ids"], 
                packed_context["packed_context"]
            )
    
            # STEP 3 & 4: Cross-Document Interaction with Document-Aware Bias
            # Ensure input matches model precision (already handled by engine init)
            sentence_vectors = sentence_vectors.to(self.device).to(self.model_dtype)
            doc_ids = doc_ids.to(self.device)
            
            fused_sentence_vectors = self.fusion_layer(sentence_vectors, doc_ids)
    
            # STEP 5: Entity Alignment
            sentences = [item[0] for item in packed_context["packed_context"]]
            entity_memory, entity_to_sents = self.entity_aligner.extract_and_align(
                sentences, 
                fused_sentence_vectors,
                packed_context_list=packed_context["packed_context"]
            )
    
            # STEP 6: Contradiction Detection
            contradiction_signals = self.contradiction_detector.detect(entity_to_sents, sentences, doc_ids)
    
            # STEP 6.5: Contradiction-Aware Interaction
            # Ensure entity memory also matches precision
            for k in entity_memory:
                entity_memory[k] = entity_memory[k].to(self.model_dtype)
    
            fused_sentence_vectors = self.interaction_layer(
                fused_sentence_vectors,
                entity_memory,
                entity_to_sents,
                contradiction_signals
            )
    
            # STEP 7: Build Fusion Memory
            # The final unified semantic memory structure. Move to CPU for storage efficiency.
            fusion_memory = {
                "cluster_id": cluster_data["cluster_id"],
                "sentence_vectors": fused_sentence_vectors.cpu(), # Move to CPU
                "entity_memory": {k: v.cpu() for k, v in entity_memory.items()},
                "contradiction_signals": contradiction_signals, # Already lists/dicts
                "doc_ids": doc_ids.cpu(),
                "metadata": {
                    "num_sentences": len(sentences),
                    "num_entities": len(entity_memory),
                    "num_contradictions": len(contradiction_signals)
                }
            }
    
            return fusion_memory

    def run_full_pipeline(self, base_cache_dir: str = "cache"):
        """
        Runs the fusion pipeline across all clusters and caches results aggressively.
        """
        from tqdm import tqdm
        fusion_dir = os.path.join(base_cache_dir, "fusion")
        entity_dir = os.path.join(base_cache_dir, "entities")
        contradiction_dir = os.path.join(base_cache_dir, "contradiction")
        
        for d in [fusion_dir, entity_dir, contradiction_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Load all packed contexts for lookup
        all_packed = self.loader.load_packed_contexts()
        context_map = {c['cluster_id']: c for c in all_packed}
        
        print(f"🚀 Starting Phase 5 Fusion Pipeline...")
        
        # Stream cluster data (Memory optimized: Do NOT convert to list)
        # We estimate total for tqdm by counting files in shards or output dir
        total_clusters = 0
        if os.path.exists(self.loader.shard_dir):
            # Each shard has SHARD_SIZE clusters (usually 100)
            shards = [f for f in os.listdir(self.loader.shard_dir) if f.endswith(".pt")]
            total_clusters = len(shards) * 100 
        elif os.path.exists(self.loader.output_dir):
            total_clusters = len([f for f in os.listdir(self.loader.output_dir) if f.endswith(".pt")])
            
        for i, cluster_data in enumerate(tqdm(self.loader.stream_data(), total=total_clusters, desc="Fusion Progress")):
            cluster_id = cluster_data["cluster_id"]
            
            if cluster_id not in context_map:
                continue
                
            packed_context = context_map[cluster_id]
            
            # Resume Logic: Skip if already processed
            final_path = os.path.join(fusion_dir, f"{cluster_id}.pt")
            if os.path.exists(final_path):
                continue
                
            # Log progress
            tqdm.write(f"📂 Processing Cluster {cluster_id}...")
                
            # Execute Pipeline
            fusion_memory = self.fuse_cluster(cluster_data, packed_context)
            
            # STEP 8: SAVE EVERYTHING (Aggressive Caching)
            torch.save(fusion_memory, os.path.join(fusion_dir, f"{cluster_id}.pt"))
            torch.save(fusion_memory["entity_memory"], os.path.join(entity_dir, f"{cluster_id}_entities.pt"))
            torch.save(fusion_memory["contradiction_signals"], os.path.join(contradiction_dir, f"{cluster_id}_conflicts.pt"))
            
            # Memory Management: Clear GPU cache after every cluster for Colab stability
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        print(f"✅ Phase 5 complete. All artifacts cached in {base_cache_dir}/")

if __name__ == "__main__":
    # Example usage (will run if data is present)
    engine = FusionEngine()
    # engine.run_full_pipeline()
