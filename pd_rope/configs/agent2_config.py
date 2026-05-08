"""Central configuration for Agent 2 (Cross-Document Aggregator)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PDRoPEConfig:
    """Hyper-parameters for the PD-RoPE positional encoding layer."""
    dim: int = 768
    max_doc_len: int = 32
    max_sent_len: int = 512
    doc_base: float = 1_000.0
    sent_base: float = 10_000.0


@dataclass
class InputConfig:
    """Configuration for the input interface with Agent 1."""
    max_seq_len: int = 100
    embed_dim: int = 768
    max_docs: int = 10


@dataclass
class Agent2Config:
    """Top-level configuration for Agent 2."""
    input: InputConfig = field(default_factory=InputConfig)
    pd_rope: PDRoPEConfig = field(default_factory=PDRoPEConfig)
    raw_data_dir: str = "data/raw"
    mock_agent1_dir: str = "data/mock_agent1"
    seed: int = 42
