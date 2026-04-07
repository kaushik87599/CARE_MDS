"""
configs/agent2_config.py
=========================
Central configuration for Agent 2 (Cross-Document Aggregation Agent).
All hyper-parameters live here so they are easy to tune without touching model code.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PDRoPEConfig:
    """Hyper-parameters for the PD-RoPE positional encoding layer."""

    # Total embedding dimension (must be divisible by 4)
    dim: int = 768

    # Maximum number of distinct documents in one packed sequence
    max_doc_len: int = 32

    # Maximum sentences per document
    max_sent_len: int = 512

    # Frequency base for the document-level axis.
    # Smaller → slower frequency decay → better doc-boundary discrimination.
    doc_base: float = 1_000.0

    # Frequency base for the sentence-level axis (standard RoPE default).
    sent_base: float = 10_000.0


@dataclass
class InputConfig:
    """Configuration for the input interface with Agent 1."""

    # Maximum total sentences in one packed multi-doc sequence (N in docs)
    max_seq_len: int = 100

    # Embedding dimension produced by Agent 1 / the encoder (BERT-base → 768)
    embed_dim: int = 768

    # Maximum documents per multi-doc cluster
    max_docs: int = 10


@dataclass
class Agent2Config:
    """Top-level configuration for Agent 2."""

    # Sub-configs
    input: InputConfig = field(default_factory=InputConfig)
    pd_rope: PDRoPEConfig = field(default_factory=PDRoPEConfig)

    # Data paths
    raw_data_dir: str = "data/raw"
    mock_agent1_dir: str = "data/mock_agent1"

    # Random seed for reproducibility
    seed: int = 42
