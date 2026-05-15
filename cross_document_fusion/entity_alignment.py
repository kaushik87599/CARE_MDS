import torch
import spacy
from typing import List, Dict, Any, Optional, Tuple

class EntityAligner:
    """
    Step 5: Entity Alignment Layer.
    Groups sentence representations by mentioned entities to create 
    shared semantic representations of key actors/concepts across documents.
    """
    def __init__(self, spacy_model: str = "en_core_web_md"):
        try:
            # Optimize with GPU if available
            spacy.prefer_gpu()
            self.nlp = spacy.load(spacy_model)
        except Exception:
            print(f"Warning: Could not load {spacy_model}, falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_and_align(self, sentences: List[str], sentence_vectors: torch.Tensor, packed_context_list: Optional[List[Any]] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[int]]]:
        """
        Creates an entity memory. Prioritizes pre-extracted entities if available,
        otherwise falls back to live NER.
        
        Returns:
            Tuple of (entity_memory, entity_to_sent_indices).
        """
        entity_to_vectors = {}
        entity_to_sent_indices = {}
        
        # 1. Attempt to reuse pre-extracted entities from packed_context
        pre_extracted = False
        if packed_context_list and len(packed_context_list) > 0:
            sample = packed_context_list[0]
            # Check if item has entities (assuming index 4 or 'entities' key)
            if (isinstance(sample, (list, tuple)) and len(sample) >= 5) or (isinstance(sample, dict) and 'entities' in sample):
                pre_extracted = True
                print("♻️ Reusing pre-extracted entities from context.")

        if pre_extracted:
            for i, item in enumerate(packed_context_list):
                # Extract entities from the item
                ents = item[4] if isinstance(item, (list, tuple)) else item.get('entities', [])
                for ent_text, ent_label in ents:
                    if ent_label in ["PERSON", "ORG", "GPE", "EVENT", "FAC"]:
                        name = ent_text.lower().strip()
                        if name not in entity_to_vectors:
                            entity_to_vectors[name] = []
                            entity_to_sent_indices[name] = []
                        entity_to_vectors[name].append(sentence_vectors[i])
                        entity_to_sent_indices[name].append(i)
        else:
            # 2. Fallback: Run live NER if no pre-extracted entities found
            for i, doc in enumerate(self.nlp.pipe(sentences)):
                seen_in_this_sentence = set()
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "FAC"]:
                        name = ent.text.lower().strip()
                        if name not in seen_in_this_sentence:
                            if name not in entity_to_vectors:
                                entity_to_vectors[name] = []
                                entity_to_sent_indices[name] = []
                            entity_to_vectors[name].append(sentence_vectors[i])
                            entity_to_sent_indices[name].append(i)
                            seen_in_this_sentence.add(name)

        # Heuristic Alignment Logic
        entity_memory = {}
        aligned_sent_indices = {}
        sorted_names = sorted(entity_to_vectors.keys(), key=len, reverse=True)
        processed = set()
        
        for name in sorted_names:
            if name in processed: continue
            
            variants = [v for v in sorted_names if (v in name or name in v) and v not in processed]
            
            all_vectors = []
            all_indices = []
            for v in variants:
                all_vectors.extend(entity_to_vectors[v])
                all_indices.extend(entity_to_sent_indices[v])
                processed.add(v)
            
            if all_vectors:
                entity_memory[name] = torch.stack(all_vectors).mean(dim=0)
                aligned_sent_indices[name] = list(set(all_indices))
        
        return entity_memory, aligned_sent_indices

    def apply_entity_bias(self, sentence_vectors: torch.Tensor, sentences: List[str], entity_memory: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        (Optional Addition) 
        Updates sentence vectors with entity-level knowledge.
        """
        # This can be used in future steps to fuse the entity representations back into sentences
        pass

if __name__ == "__main__":
    print("EntityAligner implementation complete.")
