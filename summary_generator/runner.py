import os
import torch
import json
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from decoder.memory_loader import MemoryLoader
from decoder.state_preparer import DecoderStatePreparer
from summary_generator.generator import SummaryGenerator
from summary_generator.evaluator import SummaryEvaluator

# Load environment variables
load_dotenv()

class Phase7Runner:
    """
    Phase 7 Orchestrator: Summary Generation & Evaluation.
    Integrates memory loading, state preparation, controlled decoding, and multi-metric evaluation.
    """
    def __init__(self, 
                 fusion_dir: str = None, 
                 output_dir: str = None,
                 model_path: str = None,
                 do_eval: bool = False):
        
        # Use centrally managed paths from environment
        self.fusion_dir = fusion_dir or os.getenv("FUSION_DIR", "cache/fusion")
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "outputs/generated_summaries")
        
        if model_path is None:
            models_dir = os.getenv("MODELS_DIR", "models")
            model_path = os.path.join(models_dir, "final_mds_led")
            
        os.makedirs(self.output_dir, exist_ok=True)
        self.do_eval = do_eval
        
        # Initialize components
        self.loader = MemoryLoader(base_dir=self.fusion_dir)
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
            # 1. Load Fused Semantic Memory
            memory = self.loader.load_memory(cluster_id)
            fused_sentence_vectors = memory["fused_sentence_vectors"]
            contradiction_memory = memory.get("contradiction_memory", [])
            entity_memory = memory.get("entity_embeddings", {}) or memory.get("entity_memory", {})
            
            # Determine if hedging is needed
            has_contradictions = len(contradiction_memory) > 0
            
            # 2. Reconstruct Decoder-Compatible Encoder States
            fused_hidden_states, fused_attention_mask = self.preparer.prepare_states(fused_sentence_vectors)
            
            # 3. Generate Summary
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
            
            # 4. Evaluation
            eval_results = None
            if self.do_eval and reference_summary and source_sentences:
                print(f"⚖️ Running evaluation for {cluster_id}...")
                eval_results = self.evaluator.evaluate(
                    summary=summary_text,
                    reference=reference_summary,
                    source_sentences=source_sentences,
                    contradictions=contradiction_memory,
                    entities=list(entity_memory.keys())
                )

            # 5. Save Outputs
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
                
            print(f"✅ Summary saved for {cluster_id}")
            return summary_text
            
        except Exception as e:
            print(f"❌ Error generating summary for cluster {cluster_id}: {e}")
            return None

    def run_all(self):
        """
        Runs the summary generation pipeline for all available fusion artifacts.
        """
        if not os.path.exists(self.fusion_dir):
            print(f"❌ Fusion directory {self.fusion_dir} does not exist.")
            return

        cluster_files = sorted([f for f in os.listdir(self.fusion_dir) if f.endswith(".pt")])
        cluster_ids = [f.replace(".pt", "") for f in cluster_files]
        
        # Resume Logic
        existing_results = [cid for cid in cluster_ids if os.path.exists(os.path.join(self.output_dir, f"{cid}_results.json"))]
        last_cid = existing_results[-1] if existing_results else None
        
        print(f"🚀 Starting Phase 7: Processing {len(cluster_ids)} clusters...")
        
        for cluster_id in tqdm(cluster_ids, desc="Generating Summaries"):
            if os.path.exists(os.path.join(self.output_dir, f"{cluster_id}_results.json")) and cluster_id != last_cid:
                continue
            self.run_for_cluster(cluster_id)
            
        print(f"🏁 Phase 7 Complete.")

if __name__ == "__main__":
    runner = Phase7Runner(do_eval=False)
    # runner.run_all()
