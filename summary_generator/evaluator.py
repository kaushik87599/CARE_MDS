import torch
import evaluate
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any

class SummaryEvaluator:
    """
    Phase 7 Step 8: Multi-Metric Evaluation Pipeline.
    Evaluates beyond ROUGE to focus on reasoning, consistency, and factuality.
    """
    def __init__(self, nlp_model: str = "en_core_web_md"):
        print("🧪 Initializing Evaluation Pipeline...")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.nlp = spacy.load(nlp_model)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Evaluator ready on {self.device}")

    def evaluate(self, 
                 summary: str, 
                 reference: str, 
                 source_sentences: List[str], 
                 contradictions: List[Dict],
                 entities: List[str]) -> Dict[str, Any]:
        """
        Runs the full evaluation suite for a single summary.
        """
        results = {}

        # 1. ROUGE (Standard overlap)
        rouge_results = self.rouge.compute(predictions=[summary], references=[reference])
        results["rouge"] = rouge_results

        # 2. BERTScore (Semantic overlap)
        bs_results = self.bertscore.compute(predictions=[summary], references=[reference], lang="en", device=str(self.device))
        results["bertscore"] = {
            "precision": np.mean(bs_results["precision"]),
            "recall": np.mean(bs_results["recall"]),
            "f1": np.mean(bs_results["f1"])
        }

        # 3. Coherence (Adjacent sentence similarity)
        results["coherence"] = self._calculate_coherence(summary)

        # 4. Entity Consistency (Percentage of input entities preserved)
        results["entity_consistency"] = self._check_entity_consistency(summary, entities)

        # 5. Contradiction Preservation (Hedging check)
        results["contradiction_score"] = self._check_contradiction_handling(summary, contradictions)

        # 6. Factuality (Semantic grounding in source)
        results["factuality"] = self._calculate_factuality(summary, source_sentences)

        return results

    def _calculate_coherence(self, text: str) -> float:
        """Measures coherence as the average cosine similarity between adjacent sentences."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        if len(sentences) < 2:
            return 1.0
        
        embeddings = self.embedder.encode(sentences, convert_to_tensor=True)
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
            similarities.append(sim)
            
        return np.mean(similarities)

    def _check_entity_consistency(self, summary: str, input_entities: List[str]) -> float:
        """Calculates what percentage of key input entities are preserved in the summary."""
        if not input_entities:
            return 1.0
        
        summary_doc = self.nlp(summary)
        summary_entities = set([ent.text.lower() for ent in summary_doc.ents])
        
        matches = 0
        for ent in input_entities:
            if ent.lower() in summary_entities or any(ent.lower() in s_ent for s_ent in summary_entities):
                matches += 1
                
        return matches / len(input_entities)

    def _check_contradiction_handling(self, summary: str, contradictions: List[Dict]) -> Dict[str, Any]:
        """Checks if contradictions are handled with hedging language and entities involved are mentioned."""
        if not contradictions:
            return {"status": "no_contradictions", "score": 1.0, "contradiction_recall": 1.0}
            
        hedging_terms = ["suggest", "disagree", "conflict", "however", "although", "unclear", "disputed", "reports"]
        summary_lower = summary.lower()
        
        found_terms = [term for term in hedging_terms if term in summary_lower]
        
        # Check how many detected conflicts are mentioned (at least one entity from the conflict is in the summary)
        addressed_conflicts = 0
        for conflict in contradictions:
            entities = conflict.get("entities", [])
            if any(ent.lower() in summary_lower for ent in entities):
                addressed_conflicts += 1
                
        recall = addressed_conflicts / len(contradictions)
        
        # We expect at least one hedging term if contradictions exist
        score = min(1.0, len(found_terms) / 2.0) if found_terms else 0.0
        
        return {
            "status": "contradictions_found",
            "hedging_score": score,
            "hedging_terms_found": found_terms,
            "contradiction_recall": recall
        }

    def _calculate_factuality(self, summary: str, source_sentences: List[str]) -> float:
        """Measures factuality as the average max-similarity of summary sentences to source sentences."""
        summary_doc = self.nlp(summary)
        summary_sents = [sent.text.strip() for sent in summary_doc.sents]
        
        if not summary_sents or not source_sentences:
            return 1.0
            
        summary_embeds = self.embedder.encode(summary_sents, convert_to_tensor=True)
        source_embeds = self.embedder.encode(source_sentences, convert_to_tensor=True)
        
        cos_sim = util.cos_sim(summary_embeds, source_embeds)
        max_sims = torch.max(cos_sim, dim=1).values
        
        return torch.mean(max_sims).item()

if __name__ == "__main__":
    # Small test
    evaluator = SummaryEvaluator()
    print("Evaluator module loaded.")
