import os
import torch
import json
from tqdm import tqdm
from datetime import datetime
from decoder.memory_loader import MemoryLoader
from decoder.state_preparer import DecoderStatePreparer
from summary_generator.generator import SummaryGenerator
from summary_generator.evaluator import SummaryEvaluator

class Phase7Runner:
    """
    Phase 7 Orchestrator: Summary Generation & Evaluation.
    Integrates memory loading, state preparation, controlled decoding, and multi-metric evaluation.
    """
    def __init__(self, 
                 fusion_dir: str = os.getenv("FUSION_DIR", "cache/fusion"), 
                 output_dir: str = os.getenv("OUTPUT_DIR", "outputs/generated_summaries"),
                 model_path: str = os.getenv("MODEL_PATH", "models/models/final_mds_led"),
                 do_eval: bool = False):
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.do_eval = do_eval
        
        # Initialize components
        self.loader = MemoryLoader(base_dir=fusion_dir)
        self.preparer = DecoderStatePreparer(max_length=1024)
        self.generator = SummaryGenerator(model_path=model_path)
        
        if self.do_eval:
            self.evaluator = SummaryEvaluator()
        
        print(f"🌟 Phase 7 Runner initialized. Storing results in: {self.output_dir}")

    def run_for_cluster(self, cluster_id: str, reference_summary: str = None, source_sentences: list = None):
        """
        Processes a single cluster to generate its summary, evaluate it, and save all metadata.
        """
        try:
            # 1. Load Fused Semantic Memory (Step 1)
            memory = self.loader.load_memory(cluster_id)
            fused_sentence_vectors = memory["fused_sentence_vectors"]
            contradiction_memory = memory.get("contradiction_memory", [])
            entity_memory = memory.get("entity_embeddings", {}) or memory.get("entity_memory", {})
            
            # STEP 6: Determine if hedging is needed
            has_contradictions = len(contradiction_memory) > 0
            
            # 2. Reconstruct Decoder-Compatible Encoder States (Step 2)
            fused_hidden_states, fused_attention_mask = self.preparer.prepare_states(fused_sentence_vectors)
            
            # 3, 4, 5, 6. Inject Memory and Generate Summary
            generation_config = {
                "max_length": 256,
                "min_length": 64,
                "num_beams": 4,
                "repetition_penalty": 2.0,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "apply_hedging": has_contradictions
            }
            
            summary_text = self.generator.generate_summary(
                fused_hidden_states, 
                fused_attention_mask,
                **generation_config
            )
            
            # STEP 8: Evaluation
            eval_results = None
            if self.do_eval and reference_summary and source_sentences:
                print(f"⚖️ Running multi-metric evaluation for {cluster_id}...")
                eval_results = self.evaluator.evaluate(
                    summary=summary_text,
                    reference=reference_summary,
                    source_sentences=source_sentences,
                    contradictions=contradiction_memory,
                    entities=list(entity_memory.keys())
                )

            # STEP 7: Save Outputs
            output_data = {
                "cluster_id": cluster_id,
                "generated_summary": summary_text,
                "generation_config": generation_config,
                "contradiction_metadata": {
                    "count": len(contradiction_memory),
                    "signals": contradiction_memory
                },
                "entity_metadata": {
                    "count": len(entity_memory),
                    "entities": list(entity_memory.keys())
                },
                "evaluation": eval_results,
                "timestamp": datetime.now().isoformat()
            }
            
            output_path = os.path.join(self.output_dir, f"{cluster_id}_results.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)
            
            # Also save a plain text version
            txt_path = os.path.join(self.output_dir, f"{cluster_id}_summary.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
                
            print(f"✅ Summary and metadata saved for {cluster_id} -> {output_path}")
            return summary_text
            
        except Exception as e:
            print(f"❌ Error generating summary for cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all(self):
        """
        Runs the summary generation pipeline for all available fusion artifacts.
        """
        if not os.path.exists(self.loader.base_dir):
            print(f"❌ Fusion directory {self.loader.base_dir} does not exist.")
            return

        cluster_files = sorted([f for f in os.listdir(self.loader.base_dir) if f.endswith(".pt")])
        cluster_ids = [f.replace(".pt", "") for f in cluster_files]
        
        # Persistence Logic: Identify existing results to resume
        existing_results = [cid for cid in cluster_ids if os.path.exists(os.path.join(self.output_dir, f"{cid}_results.json"))]
        last_cid = existing_results[-1] if existing_results else None
        
        if last_cid:
            print(f"🔄 Found {len(existing_results)} existing summaries. Last reached: {last_cid}")
            print(f"⏩ Skipping already processed clusters, but will re-generate {last_cid} to ensure integrity.")
        
        print(f"🚀 Starting Phase 7: Processing {len(cluster_ids)} clusters...")
        
        for cluster_id in tqdm(cluster_ids, desc="Generating Summaries"):
            # Skip if results already exist, but NOT if it's the last one found 
            # (it might be corrupted or incomplete from the previous run)
            if os.path.exists(os.path.join(self.output_dir, f"{cluster_id}_results.json")) and cluster_id != last_cid:
                continue
                
            self.run_for_cluster(cluster_id)
            
        print(f"🏁 Phase 7 Complete. All outputs cached in {self.output_dir}")

if __name__ == "__main__":
    runner = Phase7Runner(do_eval=False) # Evaluation requires references
    # runner.run_all()
