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
        self.representer = SentenceRepresenter(tokenizer_path)
        self.fusion_layer = CrossDocumentFusionLayer()
        self.entity_aligner = EntityAligner()
        self.contradiction_detector = ContradictionDetector()
        self.interaction_layer = FusionInteraction()

    def fuse_cluster(self, cluster_data: Dict[str, Any], packed_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single cluster through the entire Phase 6 pipeline.
        
        Args:
            cluster_data: Dict containing hidden_states, input_ids from Phase 5.
            packed_context: Dict containing original sentences and metadata from Phase 4.
            
        Returns:
            fusion_memory: The reasoning substrate for Phase 7.
        """
        # STEP 2 & 4: Recover Sentence-Level Representations and Document IDs
        sentence_vectors, doc_ids = self.representer.recover_vectors(
            cluster_data["hidden_states"], 
            cluster_data["input_ids"], 
            packed_context["packed_context"]
        )

        # STEP 3 & 4: Cross-Document Interaction with Document-Aware Bias
        # This allows sentences from different docs to "talk" to each other.
        fused_sentence_vectors = self.fusion_layer(sentence_vectors, doc_ids)

        # STEP 5: Entity Alignment
        # Groups information around specific actors/concepts across documents.
        sentences = [item[0] for item in packed_context["packed_context"]]
        entity_memory, entity_to_sents = self.entity_aligner.extract_and_align(
            sentences, 
            fused_sentence_vectors,
            packed_context_list=packed_context["packed_context"]
        )

        # STEP 6: Contradiction Detection
        # Identifies conflicting claims across documents for the same entities.
        contradiction_signals = self.contradiction_detector.detect(entity_to_sents, sentences, doc_ids)

        # STEP 6.5: Contradiction-Aware Interaction (Interaction & Context Fusion)
        # Refines the vectors using entity memory and marks contradictions.
        fused_sentence_vectors = self.interaction_layer(
            fused_sentence_vectors,
            entity_memory,
            entity_to_sents,
            contradiction_signals
        )

        # STEP 7: Build Fusion Memory
        # The final unified semantic memory structure.
        fusion_memory = {
            "cluster_id": cluster_data["cluster_id"],
            "sentence_vectors": fused_sentence_vectors, # Now Interaction-Aware
            "entity_memory": entity_memory,
            "contradiction_signals": contradiction_signals,
            "doc_ids": doc_ids,
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
        
        Directory Structure:
        cache/fusion/         -> Full fusion memory
        cache/entities/       -> Entity-specific representations
        cache/contradiction/  -> Detected conflicts
        """
        fusion_dir = os.path.join(base_cache_dir, "fusion")
        entity_dir = os.path.join(base_cache_dir, "entities")
        contradiction_dir = os.path.join(base_cache_dir, "contradiction")
        
        for d in [fusion_dir, entity_dir, contradiction_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Load all packed contexts for lookup
        all_packed = self.loader.load_packed_contexts()
        context_map = {c['cluster_id']: c for c in all_packed}
        
        print(f"🚀 Starting Phase 6 Fusion Pipeline...")
        
        # Stream cluster data (Memory optimized)
        for cluster_data in self.loader.stream_data():
            cluster_id = cluster_data["cluster_id"]
            
            if cluster_id not in context_map:
                continue
                
            packed_context = context_map[cluster_id]
            
            # Execute Phase 6 Pipeline
            fusion_memory = self.fuse_cluster(cluster_data, packed_context)
            
            # STEP 8: SAVE EVERYTHING (Aggressive Caching)
            # 1. Save main fusion memory
            torch.save(fusion_memory, os.path.join(fusion_dir, f"{cluster_id}.pt"))
            
            # 2. Save entity memory separately for inspection/RL
            torch.save(fusion_memory["entity_memory"], os.path.join(entity_dir, f"{cluster_id}_entities.pt"))
            
            # 3. Save contradiction signals separately
            torch.save(fusion_memory["contradiction_signals"], os.path.join(contradiction_dir, f"{cluster_id}_conflicts.pt"))
            
        print(f"✅ Phase 6 complete. All artifacts cached in {base_cache_dir}/")

if __name__ == "__main__":
    # Example usage (will run if data is present)
    engine = FusionEngine()
    # engine.run_full_pipeline()
