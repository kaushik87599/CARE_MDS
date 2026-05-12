# CARE_MDS: Technical Documentation

This document specifies the low-level technical implementation details, tensor dimensionalities, memory management strategies, and precise input/output structures across all phases of the CARE_MDS pipeline.

---

## Execution Environment Overview
- **Hardware Profile**: Designed primarily for constrained GPU environments (e.g., Google Colab T4 with ~15GB VRAM) but scales with hardware.
- **Frameworks**: PyTorch (with `torch.inference_mode()` utilized aggressively to save memory gradients), Hugging Face Transformers.
- **Base Models**: `allenai/led-large-16384` for core encoding and generation. Fine-tuned SpaCy/Transformer models for NER.

---

## Phase 1: Dataset Preprocessing

### Technical Specs & Environment
- **Memory Management**: Uses Pandas/Hugging Face `datasets` optimized loading. Uses generator-based batching for tokenization and NER to avoid RAM spikes.
- **Max Token Lengths**: Prepared with LED capacity in mind (up to 16,384 tokens), though practical subsets may be constrained down to 4,096 tokens during the encoder phase.

### Input/Output
- **Input Location**: `datasets/` (Hugging Face or raw CSV/JSON formats).
- **Output Locations**: 
  - Cache Base: `cache/`
  - Model Checkpoint: `models/final_mds_led/` (Saved using `AutoModelForSeq2SeqLM` and `AutoTokenizer` persistence).
- **Output Formats**:
  - `tokenized_data`, `entities`, `embeddings`: Pickled/structured data caches depending on `save_cache` utility.
  - Metrics: Saved natively or printed out during runtime.

---

## Phase 2: Core Encoder

### Technical Specs & Environment
- **Execution Mode**: `torch.inference_mode()` enforced. FP16 (Mixed Precision) often utilized implicitly by hardware loading.
- **Memory Management**: 
  - **Batch Size**: strictly set to `4` (conservative limit mapped to ~15GB VRAM).
  - `torch.cuda.empty_cache()` invoked at end-of-pipeline.
- **Tensor Dimensionalities**:
  - `input_ids`: `(batch_size, 4096)` (Truncated & Padded to `max_length=4096`).
  - `attention_mask`: `(batch_size, 4096)`.
  - `global_attention_mask`: `(batch_size, 4096)` where `[:, 0] = 1`.
  - `batch_hidden_states`: `(batch_size, 4096, 1024)` (LED-large hidden dim is `1024`).
- **Sharding Logic**: Slices the batch tensor `[i:i+1]` to maintain the 3D shape `(1, 4096, 1024)` before pushing to `.cpu()` for disk saving.

### Input/Output
- **Input Location**: `packed_contexts` data structures (loaded via internal `loader.py`).
- **Output Location**: `cache/encoder_outputs/{cluster_id}.pt`.
- **Output Format**: PyTorch dictionary containing:
  ```python
  {
      "cluster_id": str,
      "hidden_states": Tensor(1, 4096, 1024), # Pushed to CPU
      "attention_mask": Tensor(1, 4096),      # Pushed to CPU
      "input_ids": Tensor(1, 4096)            # Pushed to CPU
  }
  ```

---

## Phase 3: Cross-Document Fusion

### Technical Specs & Environment
- **Execution Mode**: Streams data iteratively per-cluster (`self.loader.stream_data()`) to prevent RAM/VRAM exhaustion from bulk loading thousands of documents.
- **Tensor Dimensionalities**:
  - `sentence_vectors`: Dynamically sized based on sentence count -> `(num_sentences, 1024)`.
  - `fused_sentence_vectors` (Post Cross-Doc Layer): `(num_sentences, 1024)`.
  - `entity_memory` embeddings: Varies by entity count, mapped locally.
- **Memory Management**: Aggressive caching strategy. Results are flushed to disk immediately after each cluster is processed.

### Input/Output
- **Input Location**: `cache/encoder_outputs/{cluster_id}.pt` and `packed_context` metadata.
- **Output Locations & Formats**:
  - Main Fusion Memory: `cache/fusion/{cluster_id}.pt`
    ```python
    {
        "cluster_id": str,
        "sentence_vectors": Tensor(num_sentences, 1024), # Interaction-aware
        "entity_memory": dict, 
        "contradiction_signals": list/dict,
        "doc_ids": list,
        "metadata": dict(num_sentences, num_entities, num_contradictions)
    }
    ```
  - Sub-artifacts for inspection/RL tracking:
    - Entities: `cache/entities/{cluster_id}_entities.pt`
    - Conflicts: `cache/contradiction/{cluster_id}_conflicts.pt`

---

## Phase 4: Decoding State Preparation

### Technical Specs & Environment
- **Tensor Processing**: Explicit `.to(device)` mapping just-in-time for decoder execution.
- **Padding/Truncation Engine**: Forces arbitrary `num_sentences` into strict matrix constraints.
- **Tensor Dimensionalities**:
  - Input: `fused_sentence_vectors` -> `(num_sentences, 1024)` or `(1024)` which gets `unsqueeze(0)`'d.
  - Processed Truncation: Slices to `[:1024, :]` if `num_sentences > 1024`.
  - Processed Padding: `F.pad(tensor, (0, 0, 0, padding_len))` if `num_sentences < 1024`.
  - Output `fused_hidden_states`: `(1, 1024, 1024)` (Batch, Max Length, Hidden Dim).
  - Output `fused_attention_mask`: `(1, 1024)` (Populated with `1`s for real sentences, `0`s for pads).

---

## Phase 5: Summary Generation

### Technical Specs & Environment
- **Execution Mode**: Generative loop running on GPU.
- **Decoding Configuration**:
  - `max_length`: 256
  - `min_length`: 64
  - `num_beams`: 4
  - `repetition_penalty`: 2.0
  - `length_penalty`: 1.0
  - `no_repeat_ngram_size`: 3
  - `early_stopping`: True
  - `apply_hedging`: Dynamic Boolean based on `len(contradiction_memory) > 0`.

### Input/Output
- **Input Location**: 
  - State Memory: `cache/fusion/{cluster_id}.pt`.
  - Model weights: `models/models/final_mds_led`
- **Output Locations**: 
  - `outputs/generated_summaries/`
- **Output Formats**:
  - `.txt` File: Pure string representation of the generated summary (`{cluster_id}_summary.txt`).
  - `.json` File: Comprehensive metadata payload (`{cluster_id}_results.json`):
    ```json
    {
        "cluster_id": "string",
        "generated_summary": "string",
        "generation_config": { ... },
        "contradiction_metadata": { "count": int, "signals": [...] },
        "entity_metadata": { "count": int, "entities": [...] },
        "evaluation": { ... }, 
        "timestamp": "ISO-8601 string"
    }
    ```
