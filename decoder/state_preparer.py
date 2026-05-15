import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class DecoderStatePreparer:
    """
    Phase 7 Step 2: Prepares fused sentence vectors for use as 
    encoder_hidden_states in the LED decoder.
    """
    def __init__(self, max_length: int = 1024, hidden_size: int = 1024):
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🛠️ DecoderStatePreparer initialized. Max sentences: {self.max_length}")

    def prepare_states(self, fused_sentence_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts fused sentence vectors into hidden states and attention mask.
        Optimized to use actual sequence length rather than fixed padding.
        """
        # 1. Take fused sentence vectors
        if len(fused_sentence_vectors.shape) == 1:
            fused_sentence_vectors = fused_sentence_vectors.unsqueeze(0)
            
        num_sentences = fused_sentence_vectors.shape[0]
        
        # 2. Handle Truncation ONLY (No padding to 1024 needed for decoder injection)
        if num_sentences > self.max_length:
            print(f"✂️ Truncating {num_sentences} sentences to {self.max_length}")
            processed_vectors = fused_sentence_vectors[:self.max_length, :]
            actual_len = self.max_length
        else:
            processed_vectors = fused_sentence_vectors
            actual_len = num_sentences

        # 3. Create fused_hidden_states (Add batch dimension)
        # Shape: (1, actual_len, 1024)
        fused_hidden_states = processed_vectors.unsqueeze(0).to(self.device)
        
        # 4. Create fused_attention_mask (All 1s for the actual vectors)
        fused_attention_mask = torch.ones((1, actual_len), dtype=torch.long, device=self.device)
        # print(f"📊 Fused Hidden States ready: {fused_hidden_states.shape}")
        # print(f"🎭 Fused Attention Mask ready: {fused_attention_mask.shape}")
        
        return fused_hidden_states, fused_attention_mask

if __name__ == "__main__":
    # Test prep
    preparer = DecoderStatePreparer(max_length=50)
    
    # Test case 1: No truncation
    dummy_vectors = torch.randn(20, 1024)
    states, mask = preparer.prepare_states(dummy_vectors)
    assert states.shape == (1, 20, 1024)
    assert mask.shape == (1, 20)
    assert torch.all(mask == 1)
    
    # Test case 2: Truncation
    dummy_vectors = torch.randn(100, 1024)
    states, mask = preparer.prepare_states(dummy_vectors)
    assert states.shape == (1, 50, 1024)
    assert mask.shape == (1, 50)
    
    print("Tests passed!")
