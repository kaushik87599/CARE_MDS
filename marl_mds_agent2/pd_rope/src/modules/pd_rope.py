"""
pd_rope.py — Positional Disentangling Rotary Positional Embeddings (PD-RoPE)
==============================================================================
Disentangles document-level and sentence-level positional signals in 
multi-document summarization.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def initialize_rotation_angles(
    seq_len: int,
    dim: int,
    base: float = 10_000.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the cos/sin tables for a single positional axis."""
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    half = dim // 2
    # θ_k = base^{-2k/dim}  →  shape (half,)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )

    # positions:  shape (seq_len,)
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)

    # outer product  →  shape (seq_len, half)
    angles = torch.einsum("i,j->ij", positions, inv_freq)

    cos_cached = angles.cos().to(dtype=dtype or torch.float32)
    sin_cached = angles.sin().to(dtype=dtype or torch.float32)
    return cos_cached, sin_cached


def half_rotate(x: torch.Tensor) -> torch.Tensor:
    """Implement the cross-pair rotation used in RoPE."""
    # x: (..., dim)  where dim is even
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotation(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation to the input tensor."""
    # Tile: (N, half) → (N, dim)
    cos_full = torch.cat([cos, cos], dim=-1)   # (N, dim)
    sin_full = torch.cat([sin, sin], dim=-1)   # (N, dim)

    # Broadcast over batch dimension
    cos_full = cos_full.unsqueeze(0)  # (1, N, dim)
    sin_full = sin_full.unsqueeze(0)  # (1, N, dim)

    return x * cos_full + half_rotate(x) * sin_full


# ---------------------------------------------------------------------------
# Core RotaryEmbedding — single-axis baseline
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Standard single-axis Rotary Positional Embedding."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Register as buffers for automatic device/dtype management
        cos_cached, sin_cached = initialize_rotation_angles(
            max_seq_len, dim, base=base
        )
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    # ------------------------------------------------------------------
    def expand_cache_if_needed(self, seq_len: int, device: torch.device) -> None:
        """Lazily expand cache if input exceeds current size."""
        if seq_len > self.cos_cached.shape[0]:
            cos_cached, sin_cached = initialize_rotation_angles(
                seq_len, self.dim, base=self.base,
                device=device, dtype=self.cos_cached.dtype
            )
            self.cos_cached = cos_cached
            self.sin_cached = sin_cached

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply RoPE to embedding `x`."""
        B, N, D = x.shape
        self.expand_cache_if_needed(N, x.device)

        if positions is None:
            cos = self.cos_cached[:N]           # (N, dim//2)
            sin = self.sin_cached[:N]           # (N, dim//2)
        else:
            # positions shape: (N,) or (B, N)
            if positions.dim() == 2:
                # Use only the first row for indexing (assumes same positions
                # across batch); full batch-aware indexing handled in PDRoPE.
                positions = positions[0]
            cos = self.cos_cached[positions]    # (N, dim//2)
            sin = self.sin_cached[positions]    # (N, dim//2)

        return apply_rotation(x, cos, sin)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"


# ---------------------------------------------------------------------------
# PD-RoPE — Positional Disentangling Rotary Positional Embeddings
# ---------------------------------------------------------------------------

class PDRoPE(nn.Module):
    """Positional Disentangling Rotary Positional Embeddings (PD-RoPE)."""

    def __init__(
        self,
        dim: int,
        max_doc_len: int = 32,
        max_sent_len: int = 512,
        doc_base: float = 1_000.0,
        sent_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(
                f"PDRoPE requires dim divisible by 4 (got {dim}). "
                "Each positional axis needs an even sub-dimension."
            )

        self.dim = dim
        self.dim_doc = dim // 2      # Half of embed dim → doc axis
        self.dim_sent = dim - self.dim_doc  # Other half → sent axis (== dim//2)

        self.doc_rope = RotaryEmbedding(
            dim=self.dim_doc,
            max_seq_len=max_doc_len,
            base=doc_base,
        )
        self.sent_rope = RotaryEmbedding(
            dim=self.dim_sent,
            max_seq_len=max_sent_len,
            base=sent_base,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        doc_ids: torch.Tensor,
        sent_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply PD-RoPE to a batch of sentence embeddings."""
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(
                f"Input embedding dim {D} does not match PDRoPE dim {self.dim}."
            )

        # Normalise position index shapes → (N,)
        doc_pos = doc_ids[0] if doc_ids.dim() == 2 else doc_ids    # (N,)
        sent_pos = sent_ids[0] if sent_ids.dim() == 2 else sent_ids  # (N,)

        # Validate ranges
        max_doc = doc_pos.max().item()
        max_sent = sent_pos.max().item()
        if max_doc >= self.doc_rope.cos_cached.shape[0]:
            self.doc_rope.expand_cache_if_needed(int(max_doc) + 1, x.device)
        if max_sent >= self.sent_rope.cos_cached.shape[0]:
            self.sent_rope.expand_cache_if_needed(int(max_sent) + 1, x.device)

        # Split embedding along feature axis
        x_doc = x[..., : self.dim_doc]    # (B, N, dim_doc)
        x_sent = x[..., self.dim_doc :]   # (B, N, dim_sent)

        # Retrieve cached cos/sin for each sentence's positional indices
        cos_doc = self.doc_rope.cos_cached[doc_pos]    # (N, dim_doc//2)
        sin_doc = self.doc_rope.sin_cached[doc_pos]    # (N, dim_doc//2)
        cos_sent = self.sent_rope.cos_cached[sent_pos] # (N, dim_sent//2)
        sin_sent = self.sent_rope.sin_cached[sent_pos] # (N, dim_sent//2)

        # Apply independent rotations
        x_doc_rotated = apply_rotation(x_doc, cos_doc, sin_doc)
        x_sent_rotated = apply_rotation(x_sent, cos_sent, sin_sent)

        # Concatenate back to full embedding
        return torch.cat([x_doc_rotated, x_sent_rotated], dim=-1)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"dim_doc={self.dim_doc}, "
            f"dim_sent={self.dim_sent}, "
            f"doc_rope=({self.doc_rope.extra_repr()}), "
            f"sent_rope=({self.sent_rope.extra_repr()})"
        )
