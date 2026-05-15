import torch
import gc
from transformers import pipeline
from typing import List, Dict, Any, Tuple, Optional

class ContradictionDetector:
    """
    Step 6: Contradiction Detection Layer.
    Detects conflicting claims across documents by comparing entity-linked sentence pairs.
    """
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        # Detect device
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Initializing NLI pipeline with {model_name} on device {self.device}...")
        
        # Load the NLI pipeline with FP16 optimization if on GPU
        model_kwargs = {"dtype": torch.float16} if self.device != -1 else {}
        
        self.nli_pipe = pipeline(
            "text-classification", 
            model=model_name, 
            device=self.device,
            top_k=None,
            model_kwargs=model_kwargs,
            truncation=True, # Ensure truncation is enabled
            max_length=512   # Deberta standard
        )
        
        # Default batch size - reduced for stability
        self.batch_size = 16

    def detect(self, entity_to_sents: Dict[str, List[int]], sentences: List[str], doc_ids: torch.Tensor, max_pairs: int = 500) -> List[Dict[str, Any]]:
        """
        Runs NLI on pairs of sentences that share entities across different documents.
        Optimized with pair deduplication, batching, and safety caps.
        """
        unique_pairs = {} # (idx1, idx2) -> List of entities
        
        # 1. Targeted Pair Selection with Deduplication
        for ent_name, sent_indices in entity_to_sents.items():
            for i in range(len(sent_indices)):
                for j in range(i + 1, len(sent_indices)):
                    idx1 = sent_indices[i]
                    idx2 = sent_indices[j]
                    
                    # Ensure consistent ordering for deduplication
                    p1, p2 = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
                    
                    # Rule: Only compare if they are from DIFFERENT documents
                    if doc_ids[p1] != doc_ids[p2]:
                        pair_key = (p1, p2)
                        if pair_key not in unique_pairs:
                            # Safety cap check
                            if len(unique_pairs) >= max_pairs:
                                break
                            unique_pairs[pair_key] = []
                        unique_pairs[pair_key].append(ent_name)
                if len(unique_pairs) >= max_pairs:
                    print(f"⚠️ Reached max_pairs cap ({max_pairs}). Skipping remaining pairs for this cluster.")
                    break

        if not unique_pairs:
            return []

        # Prepare for pipeline
        pairs_list = []
        metadata = []
        for (p1, p2), entities in unique_pairs.items():
            pairs_list.append(f"{sentences[p1]} [SEP] {sentences[p2]}")
            metadata.append({
                "entities": entities,
                "sent1_idx": p1,
                "sent2_idx": p2,
                "doc1_id": doc_ids[p1].item(),
                "doc2_id": doc_ids[p2].item(),
                "text1": sentences[p1],
                "text2": sentences[p2]
            })

        # 2. Run NLI Analysis
        print(f"Checking {len(pairs_list)} unique entity-linked pairs (Deduplicated from {sum(len(e) for e in unique_pairs.values())})...")
        
        # Memory cleanup before heavy operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        results = []
        try:
            with torch.no_grad():
                # Using a generator for the pipeline can be more memory efficient
                def data_generator():
                    for pair in pairs_list:
                        yield pair
                
                pipe_results = self.nli_pipe(data_generator(), batch_size=self.batch_size)
                results = list(pipe_results)
        except torch.OutOfMemoryError:
            print("⚠️ CUDA OOM detected in NLI pipe. Retrying with minimal batch size...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Retry with batch size 1 for maximum memory safety
            try:
                with torch.no_grad():
                    def data_generator_retry():
                        for pair in pairs_list:
                            yield pair
                    pipe_results = self.nli_pipe(data_generator_retry(), batch_size=1)
                    results = list(pipe_results)
            except torch.OutOfMemoryError:
                print("❌ Fatal CUDA OOM: Even batch size 1 failed. Skipping contradiction detection for this cluster.")
                return []
        
        contradictions = []
        
        # 3. Process and Filter Results
        for i, result in enumerate(results):
            # Deberta-v3 labels: contradiction, neutral, entailment
            # result can be a list of results if top_k is not None, 
            # but since we set top_k=None it returns a list of results for each pair
            scores = {res['label'].lower(): res['score'] for res in result}
            
            is_contradiction = scores.get('contradiction', 0) > 0.5
            
            if is_contradiction:
                entry = metadata[i].copy()
                entry['nli_scores'] = scores
                contradictions.append(entry)

        print(f"Detected {len(contradictions)} potential contradictions.")
        
        # Cleanup after operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return contradictions

if __name__ == "__main__":
    print("ContradictionDetector implementation complete.")
