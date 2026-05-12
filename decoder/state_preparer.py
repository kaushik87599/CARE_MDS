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
        Converts fused sentence vectors into padded/truncated hidden states 
        and creates the corresponding attention mask.
        
        Args:
            fused_sentence_vectors: Tensor of shape (num_sentences, 1024)
            
        Returns:
            fused_hidden_states: Tensor of shape (1, max_length, 1024)
            fused_attention_mask: Tensor of shape (1, max_length)
        """
        # 1. Take fused sentence vectors
        # Ensure it's a 2D tensor (num_sentences, hidden_size)
        if len(fused_sentence_vectors.shape) == 1:
            fused_sentence_vectors = fused_sentence_vectors.unsqueeze(0)
            
        num_sentences = fused_sentence_vectors.shape[0]
        
        # 2. Handle Padding/Truncation to fixed decoder length
        if num_sentences > self.max_length:
            # Truncate if too many sentences
            print(f"✂️ Truncating {num_sentences} sentences to {self.max_length}")
            processed_vectors = fused_sentence_vectors[:self.max_length, :]
            actual_len = self.max_length
        else:
            # Pad if fewer sentences
            padding_len = self.max_length - num_sentences
            # Pad with zeros at the end
            processed_vectors = F.pad(fused_sentence_vectors, (0, 0, 0, padding_len), value=0.0)
            actual_len = num_sentences

        # 3. Create fused_hidden_states (Add batch dimension)
        # Shape: (1, max_length, 1024)
        fused_hidden_states = processed_vectors.unsqueeze(0).to(self.device)
        
        # 4. Create fused_attention_mask
        # 1 for actual tokens (sentences), 0 for padding
        mask = torch.zeros(self.max_length, dtype=torch.long)
        mask[:actual_len] = 1
        fused_attention_mask = mask.unsqueeze(0).to(self.device)
        
        print(f"📊 Fused Hidden States ready: {fused_hidden_states.shape}")
        print(f"🎭 Fused Attention Mask ready: {fused_attention_mask.shape}")
        
        return fused_hidden_states, fused_attention_mask

if __name__ == "__main__":
    # Test prep
    preparer = DecoderStatePreparer(max_length=512)
    dummy_vectors = torch.randn(100, 1024)
    states, mask = preparer.prepare_states(dummy_vectors)
    assert states.shape == (1, 512, 1024)
    assert mask.shape == (1, 512)
    assert mask[0, 99] == 1
    assert mask[0, 100] == 0
    print("Test passed!")
