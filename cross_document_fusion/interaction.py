import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

class FusionInteraction(nn.Module):
    """
    Step 7.5: Lightweight Contradiction-Aware Interaction.
    Refines sentence vectors by injecting entity context and 
    marking contradiction signals.
    """
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Lightweight gate for entity injection
        self.entity_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Contradiction embedding (learnable bias)
        self.conflict_bias = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, 
                sentence_vectors: torch.Tensor, 
                entity_memory: Dict[str, torch.Tensor], 
                entity_to_sents: Dict[str, List[int]],
                contradictions: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Args:
            sentence_vectors: (N, 1024)
            entity_memory: {name: vector(1024)}
            entity_to_sents: {name: [indices]}
            contradictions: List of dicts with 'sent1_idx', 'sent2_idx'
            
        Returns:
            interacted_vectors: (N, 1024)
        """
        num_sentences = sentence_vectors.size(0)
        device = sentence_vectors.device
        
        # 1. Gated Entity Injection
        # We need to know which entities belong to which sentence
        # (Reversing the entity_to_sents mapping)
        sent_to_entities = [[] for _ in range(num_sentences)]
        for name, indices in entity_to_sents.items():
            for idx in indices:
                sent_to_entities[idx].append(entity_memory[name])
                
        refined_vectors = []
        for i in range(num_sentences):
            s_vec = sentence_vectors[i]
            
            if sent_to_entities[i]:
                # Aggregate context from all entities in this sentence
                entity_context = torch.stack(sent_to_entities[i]).mean(dim=0)
                
                # Compute gate based on [sentence; entity_context]
                gate_input = torch.cat([s_vec, entity_context], dim=-1)
                gate = self.entity_gate(gate_input)
                
                # Injected representation
                s_vec = s_vec + gate * entity_context
            
            refined_vectors.append(s_vec)
            
        refined_vectors = torch.stack(refined_vectors)
        
        # 2. Contradiction-Aware Weighting
        # Identify sentences involved in contradictions
        conflicting_indices = set()
        for c in contradictions:
            conflicting_indices.add(c['sent1_idx'])
            conflicting_indices.add(c['sent2_idx'])
            
        # Apply lightweight conflict bias to mark these sentences
        for idx in conflicting_indices:
            # We add a subtle signal to these vectors so the decoder 
            # knows they contain "spicy" (conflicting) information.
            refined_vectors[idx] = refined_vectors[idx] + self.conflict_bias
            
        return refined_vectors

if __name__ == "__main__":
    print("FusionInteraction module ready.")
