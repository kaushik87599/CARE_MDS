"""
pd_rope.py — Positional Disentangling Rotary Positional Embeddings (PD-RoPE)
==============================================================================
Agent 2: Cross-Document Aggregation Agent (A2) — MARL-MDS Framework

Overview
--------
Standard Rotary Positional Embeddings (RoPE) encode a single integer position
per token. In multi-document summarization, each sentence carries *two* kinds
of positional identity:

    1. p_doc  — which document (0, 1, 2, …, D-1) the sentence originates from.
    2. p_sent — the sentence's local index *within* that document (0, 1, …, N_d-1).

PD-RoPE disentangles these two positional signals by computing independent
rotation matrices for each and concatenating the resulting position-encoded
sub-spaces:

    PE(p_doc, p_sent) = R_θ(p_doc) ⊕ R_θ(p_sent)

where ⊕ denotes channel-wise concatenation over the embedding dimension.

This design gives cross-document attention two orthogonal "coordinate axes" to
reason about, enabling it to:
  - Recognise that sentences from the same document are locally adjacent.
  - Understand that two sentences at the same local index may belong to
    entirely different documents.

References
----------
Su, J. et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding."
Neurocomputing, 2024. (original RoPE)

Gemini Working Plan — Agent 2 (Cross-Doc Aggregator), Stage B.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _build_rotation_cache(
    seq_len: int,
    dim: int,
    base: float = 10_000.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute the cos/sin tables for a single positional axis.

    The rotation for a 2-D sub-space at frequency index k is:

        θ_k  = base^{-2k / dim}              (k = 0, …, dim/2 - 1)
        cos[p, k] = cos(p · θ_k)
        sin[p, k] = sin(p · θ_k)

    Parameters
    ----------
    seq_len : int
        Maximum position index to pre-compute (exclusive).
    dim : int
        Number of embedding dimensions allocated to this axis.
        Must be even.
    base : float
        Period base — larger values slow down the decay of θ,
        helping longer contexts. Default 10 000 (RoFormer default).
    device : torch.device, optional
    dtype : torch.dtype, optional

    Returns
    -------
    cos_cached : Tensor  shape (seq_len, dim // 2)
    sin_cached : Tensor  shape (seq_len, dim // 2)
    """
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the cross-pair rotation used as the 'imaginary' part of RoPE.

    For a vector split into pairs [x0, x1, x2, x3, …]:
        rotate_half → [-x1, x0, -x3, x2, …]

    This is equivalent to multiplying the complex representation by i.
    """
    # x: (..., dim)  where dim is even
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a single RoPE rotation in-place fashion.

    x_rotated = x * cos + rotate_half(x) * sin

    Convention
    ----------
    ``_rotate_half`` splits the vector at its midpoint:
        [x0, …, x_{h-1}, x_h, …, x_{2h-1}]  →  [-x_h…, x_0…]

    Therefore cos/sin, each of shape (N, h), must be tiled as
    ``cat([cos, cos])`` (same value on both halves) rather than
    interleaved, so the resulting (N, 2h) tensor aligns correctly
    with the midpoint-split rotation. This choice is what guarantees
    that the transform is orthogonal (norm-preserving).

    Parameters
    ----------
    x   : Tensor  shape (B, N, dim)
    cos : Tensor  shape (N, dim // 2)
    sin : Tensor  shape (N, dim // 2)

    Returns
    -------
    Tensor  shape (B, N, dim)
    """
    # Tile: (N, half) → (N, dim)  compatible with midpoint-split _rotate_half
    cos_full = torch.cat([cos, cos], dim=-1)   # (N, dim)
    sin_full = torch.cat([sin, sin], dim=-1)   # (N, dim)

    # Broadcast over batch dimension
    cos_full = cos_full.unsqueeze(0)  # (1, N, dim)
    sin_full = sin_full.unsqueeze(0)  # (1, N, dim)

    return x * cos_full + _rotate_half(x) * sin_full


# ---------------------------------------------------------------------------
# Core RotaryEmbedding — single-axis baseline
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Standard single-axis Rotary Positional Embedding.

    Can be used as a drop-in for vanilla RoPE within a transformer
    and also serves as the building block for PDRoPE.

    Parameters
    ----------
    dim : int
        Full embedding dimension. The rotation is applied across all `dim`
        channels (must be even).
    max_seq_len : int
        Upper bound on sequence length for cache pre-computation.
        If a sequence longer than this is encountered at runtime the cache
        is silently recomputed.
    base : float
        Frequency base (default 10 000).
    """

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

        # Register as buffers so they move with .to(device) / .to(dtype)
        cos_cached, sin_cached = _build_rotation_cache(
            max_seq_len, dim, base=base
        )
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    # ------------------------------------------------------------------
    def _maybe_extend_cache(self, seq_len: int, device: torch.device) -> None:
        """Lazily extend the cache if sequence is longer than pre-computed."""
        if seq_len > self.cos_cached.shape[0]:
            cos_cached, sin_cached = _build_rotation_cache(
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
        """
        Apply standard RoPE to embedding `x`.

        Parameters
        ----------
        x : Tensor  shape (B, N, dim)
        positions : LongTensor  shape (N,) or (B, N), optional
            Explicit position indices. If None, positions 0…N-1 are used.

        Returns
        -------
        Tensor  shape (B, N, dim) — same shape, position-rotated.
        """
        B, N, D = x.shape
        assert D == self.dim, (
            f"Embedding dim mismatch: got {D}, expected {self.dim}"
        )
        self._maybe_extend_cache(N, x.device)

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

        return _apply_rope(x, cos, sin)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"


# ---------------------------------------------------------------------------
# PD-RoPE — Positional Disentangling Rotary Positional Embeddings
# ---------------------------------------------------------------------------

class PDRoPE(nn.Module):
    """
    Positional Disentangling Rotary Positional Embeddings (PD-RoPE).

    Splits the embedding dimension into two equal halves:

        x_doc  = x[..., :dim_doc]   — rotated by document-level positions
        x_sent = x[..., dim_doc:]   — rotated by sentence-level positions

    The two halves are rotated independently and then re-concatenated:

        PE(p_doc, p_sent) = R_θ(p_doc) ⊕ R_θ(p_sent)

    Usage inside cross-document attention
    --------------------------------------
    Before computing Q and K projections, apply PDRoPE to both Q and K
    using the corresponding document IDs and sentence indices:

        Q_pos = pd_rope(Q, doc_ids, sent_ids)
        K_pos = pd_rope(K, doc_ids, sent_ids)
        scores = (Q_pos @ K_pos.transpose(-1, -2)) / sqrt(d_k)

    Parameters
    ----------
    dim : int
        Total embedding dimension. Must be divisible by 4 so that each of
        the two positional axes gets an even sub-dimension.
    max_doc_len : int
        Maximum number of distinct document IDs expected. Controls the
        size of the document-axis rotation cache.
    max_sent_len : int
        Maximum number of sentences per document. Controls the sentence-axis
        rotation cache.
    doc_base : float
        Frequency base for the document-level axis. Default 1 000.
        Keeping this smaller than sent_base ensures the document axis uses
        slower-varying frequencies — making inter-document boundaries more
        distinguishable.
    sent_base : float
        Frequency base for the sentence-level axis. Default 10 000
        (standard RoPE default).
    """

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
        """
        Apply PD-RoPE to a batch of sentence embeddings.

        Parameters
        ----------
        x : Tensor  shape (B, N, dim)
            Packed sentence embeddings from Agent 1.
        doc_ids : LongTensor  shape (N,) or (B, N)
            Document-level position index for each sentence.
            Values in range [0, max_doc_len).
        sent_ids : LongTensor  shape (N,) or (B, N)
            Sentence-level position index within its source document.
            Values in range [0, max_sent_len).

        Returns
        -------
        Tensor  shape (B, N, dim)
            Position-encoded embeddings with document and sentence axes
            disentangled in separate sub-spaces.

        Raises
        ------
        ValueError
            If `x` embedding dimension does not match `self.dim`.
        RuntimeError
            If `doc_ids` or `sent_ids` contain out-of-range indices.
        """
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
            # Extend cache lazily
            self.doc_rope._maybe_extend_cache(int(max_doc) + 1, x.device)
        if max_sent >= self.sent_rope.cos_cached.shape[0]:
            self.sent_rope._maybe_extend_cache(int(max_sent) + 1, x.device)

        # Split embedding along feature axis
        x_doc = x[..., : self.dim_doc]    # (B, N, dim_doc)
        x_sent = x[..., self.dim_doc :]   # (B, N, dim_sent)

        # Retrieve cached cos/sin for each sentence's positional indices
        cos_doc = self.doc_rope.cos_cached[doc_pos]    # (N, dim_doc//2)
        sin_doc = self.doc_rope.sin_cached[doc_pos]    # (N, dim_doc//2)
        cos_sent = self.sent_rope.cos_cached[sent_pos] # (N, dim_sent//2)
        sin_sent = self.sent_rope.sin_cached[sent_pos] # (N, dim_sent//2)

        # Apply independent rotations to each sub-space
        x_doc_rotated = _apply_rope(x_doc, cos_doc, sin_doc)
        x_sent_rotated = _apply_rope(x_sent, cos_sent, sin_sent)

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
