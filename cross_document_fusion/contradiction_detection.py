import torch
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
        
        # Load the NLI pipeline
        self.nli_pipe = pipeline(
            "text-classification", 
            model=model_name, 
            device=self.device,
            top_k=None # Returns all labels (entailment, neutral, contradiction)
        )

    def detect(self, entity_to_sents: Dict[str, List[int]], sentences: List[str], doc_ids: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Runs NLI on pairs of sentences that share entities across different documents.
        
        Args:
            entity_to_sents: Mapping of entity name to indices of sentences that mention it.
            sentences: List of actual sentence strings.
            doc_ids: Tensor of document identifiers for each sentence.
            
        Returns:
            List of detected contradictions with scores and metadata.
        """
        pairs_to_check = []
        metadata = []

        # 1. Targeted Pair Selection (Avoids computational apocalypse)
        for ent_name, sent_indices in entity_to_sents.items():
            for i in range(len(sent_indices)):
                for j in range(i + 1, len(sent_indices)):
                    idx1 = sent_indices[i]
                    idx2 = sent_indices[j]
                    
                    # Rule: Only compare if they are from DIFFERENT documents
                    if doc_ids[idx1] != doc_ids[idx2]:
                        # Combine into a single string for the cross-encoder
                        pairs_to_check.append(f"{sentences[idx1]} [SEP] {sentences[idx2]}")
                        metadata.append({
                            "entity": ent_name,
                            "sent1_idx": idx1,
                            "sent2_idx": idx2,
                            "doc1_id": doc_ids[idx1].item(),
                            "doc2_id": doc_ids[idx2].item(),
                            "text1": sentences[idx1],
                            "text2": sentences[idx2]
                        })

        if not pairs_to_check:
            return []

        # 2. Run NLI Analysis
        # Processing in batches for GPU optimization
        print(f"Checking {len(pairs_to_check)} entity-linked pairs for contradictions...")
        batch_size = 16
        results = self.nli_pipe(pairs_to_check, batch_size=batch_size)
        
        contradictions = []
        
        # 3. Process and Filter Results
        for i, result in enumerate(results):
            # result is a list of dicts: [{'label': 'CONTRADICTION', 'score': 0.9}, ...]
            scores = {res['label'].lower(): res['score'] for res in result}
            
            # Check if contradiction is the dominant label or has high confidence
            # Different models use different labels (e.g., 'contradiction', 'LABEL_0')
            # Deberta-v3-small labels: contradiction, neutral, entailment
            
            is_contradiction = scores.get('contradiction', 0) > 0.5
            
            if is_contradiction:
                entry = metadata[i].copy()
                entry['nli_scores'] = scores
                contradictions.append(entry)

        print(f"Detected {len(contradictions)} potential contradictions.")
        return contradictions

if __name__ == "__main__":
    print("ContradictionDetector implementation complete.")
