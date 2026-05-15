import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class CrossDocumentAttention(nn.Module):
    """
    Step 3: Cross-Document Attention.
    Implements scaled dot-product attention to allow interaction between 
    sentence-level representations from different documents.
    """
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        # We can add learnable projections here if we want this to be a trainable layer.
        # For now, we follow the exact formula provided: Q=K=V=sentence_vectors.
        
    def forward(self, sentence_vectors: torch.Tensor, doc_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sentence_vectors: Tensor of shape (num_sentences, 1024)
            doc_ids: Optional tensor of shape (num_sentences,) containing document identifiers.
            
        Returns:
            fused_sentence_vectors: Tensor of shape (num_sentences, 1024)
        """
        # Ensure 3D shape (batch, seq, dim)
        if sentence_vectors.dim() == 2:
            x = sentence_vectors.unsqueeze(0)
        else:
            x = sentence_vectors

        attn_mask = None
        if doc_ids is not None:
            # Step 4: Build Document-Aware Bias Matrix
            # Shape: (num_sentences, num_sentences)
            # same_doc = 1, cross_doc = 0
            doc_ids = doc_ids.to(x.device)
            relation_matrix = (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)).to(x.dtype)
            
            # For attention bias, we use this matrix. 
            # PyTorch's F.scaled_dot_product_attention adds the mask to the scores.
            # We add the relation matrix as a bias.
            attn_mask = relation_matrix.unsqueeze(0) # (1, N, N)

        # Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k) + Bias) * V
        fused = F.scaled_dot_product_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        return fused.squeeze(0) if sentence_vectors.dim() == 2 else fused

class CrossDocumentFusionLayer(nn.Module):
    """
    A more complete fusion layer that includes attention and potentially 
    other fusion components (entity alignment, contradiction handling) in future steps.
    """
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.attention = CrossDocumentAttention(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sentence_vectors: torch.Tensor, doc_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Step 3: Interaction via Cross-Document Attention
        # Now passing doc_ids to enable document-aware bias
        attended = self.attention(sentence_vectors, doc_ids)
        
        # Adding a residual connection and norm for stability
        # (Standard practice in transformer-style fusion)
        out = self.norm(sentence_vectors + attended)
        
        return out

if __name__ == "__main__":
    print("CrossDocumentAttention implementation complete.")
