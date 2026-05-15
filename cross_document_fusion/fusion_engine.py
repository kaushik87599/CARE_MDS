import torch
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from .loader import DataLoader
from .sentence_recovery import SentenceRepresenter
from .attention import CrossDocumentFusionLayer
from .entity_alignment import EntityAligner
from .contradiction_detection import ContradictionDetector
from .interaction import FusionInteraction

# Load environment variables
load_dotenv()

class FusionEngine:
    """
    Phase 6 Orchestrator: Fusion Engine.
    Combines all steps to build a unified Cross-Document Fusion Memory.
    """
    def __init__(self, tokenizer_path: str = None):
        if tokenizer_path is None:
            models_dir = os.getenv("MODELS_DIR", "models")
            tokenizer_path = os.path.join(models_dir, "final_mds_led")
            
        self.loader = DataLoader()
        self.device = self.loader.get_device()
        
        print(f"⚙️ Initializing Fusion Engine on {self.device}...")
        
        self.representer = SentenceRepresenter(tokenizer_path)
        
        # Optimization: Use Half precision for all layers if on GPU
        self.model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        self.fusion_layer = CrossDocumentFusionLayer().to(self.device).to(self.model_dtype)
        self.entity_aligner = EntityAligner()
        self.contradiction_detector = ContradictionDetector()
        self.interaction_layer = FusionInteraction().to(self.device).to(self.model_dtype)

    def fuse_cluster(self, cluster_data: Dict[str, Any], packed_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single cluster through the entire pipeline.
        """
        with torch.no_grad():
            sentence_vectors, doc_ids = self.representer.recover_vectors(
                cluster_data["hidden_states"], 
                cluster_data["input_ids"], 
                packed_context["packed_context"]
            )
    
            sentence_vectors = sentence_vectors.to(self.device).to(self.model_dtype)
            doc_ids = doc_ids.to(self.device)
            
            fused_sentence_vectors = self.fusion_layer(sentence_vectors, doc_ids)
    
            sentences = [item[0] for item in packed_context["packed_context"]]
            entity_memory, entity_to_sents = self.entity_aligner.extract_and_align(
                sentences, 
                fused_sentence_vectors,
                packed_context_list=packed_context["packed_context"]
            )
    
            contradiction_signals = self.contradiction_detector.detect(entity_to_sents, sentences, doc_ids)
    
            for k in entity_memory:
                entity_memory[k] = entity_memory[k].to(self.model_dtype)
    
            fused_sentence_vectors = self.interaction_layer(
                fused_sentence_vectors,
                entity_memory,
                entity_to_sents,
                contradiction_signals
            )
    
            fusion_memory = {
                "cluster_id": cluster_data["cluster_id"],
                "sentence_vectors": fused_sentence_vectors.cpu(),
                "entity_memory": {k: v.cpu() for k, v in entity_memory.items()},
                "contradiction_signals": contradiction_signals,
                "doc_ids": doc_ids.cpu(),
                "metadata": {
                    "num_sentences": len(sentences),
                    "num_entities": len(entity_memory),
                    "num_contradictions": len(contradiction_signals)
                }
            }
    
            return fusion_memory

    def run_full_pipeline(self):
        """
        Runs the fusion pipeline across all clusters.
        """
        from tqdm import tqdm
        fusion_dir = os.getenv("FUSION_DIR", "cache/fusion")
        entity_dir = os.getenv("ENTITY_DIR", "cache/entities")
        contradiction_dir = os.getenv("CONTRADICTION_DIR", "cache/contradiction")
        
        for d in [fusion_dir, entity_dir, contradiction_dir]:
            os.makedirs(d, exist_ok=True)
        
        all_packed = self.loader.load_packed_contexts()
        context_map = {c['cluster_id']: c for c in all_packed}
        
        print(f"🚀 Starting Phase 5 Fusion Pipeline...")
        
        total_clusters = 0
        shard_dir = os.getenv("SHARD_DIR", "cache/encoder_shards")
        encoder_out_dir = os.getenv("ENCODER_OUT_DIR", "cache/encoder_outputs")
        
        if os.path.exists(shard_dir):
            shards = [f for f in os.listdir(shard_dir) if f.endswith(".pt")]
            total_clusters = len(shards) * 100 
        elif os.path.exists(encoder_out_dir):
            total_clusters = len([f for f in os.listdir(encoder_out_dir) if f.endswith(".pt")])
            
        for i, cluster_data in enumerate(tqdm(self.loader.stream_data(), total=total_clusters, desc="Fusion Progress")):
            cluster_id = cluster_data["cluster_id"]
            
            if cluster_id not in context_map:
                continue
                
            packed_context = context_map[cluster_id]
            final_path = os.path.join(fusion_dir, f"{cluster_id}.pt")
            if os.path.exists(final_path):
                continue
                
            fusion_memory = self.fuse_cluster(cluster_data, packed_context)
            
            torch.save(fusion_memory, os.path.join(fusion_dir, f"{cluster_id}.pt"))
            torch.save(fusion_memory["entity_memory"], os.path.join(entity_dir, f"{cluster_id}_entities.pt"))
            torch.save(fusion_memory["contradiction_signals"], os.path.join(contradiction_dir, f"{cluster_id}_conflicts.pt"))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        print(f"✅ Phase 5 complete.")

if __name__ == "__main__":
    engine = FusionEngine()
    # engine.run_full_pipeline()

