# CARE_MDS: Multi-Document Summarization Workflow

This document provides a highly detailed, step-by-step workflow for the CARE_MDS (Multi-Document Summarization) pipeline. It focuses on the logical sequence of operations and the flow of data across the various phases, omitting low-level technical specifics such as tensor dimensionalities, precise hardware memory management, and specific code execution environments.

---

## Phase 1: Dataset Preprocessing (`preprocessing/`)
This initial phase ensures the input datasets are valid, fully analyzed, and transformed into a format suitable for the transformer model.

1. **Initialization & Verification**: The pipeline first downloads necessary linguistic resources (like NLTK sentence boundary markers) and verifies that the raw datasets (e.g., CNN/Daily-News, Multi-News) are present in the expected directories.
2. **Directory Setup**: It dynamically creates a clean cache directory structure based on the datasets provided.
3. **Sampling Strategy**: Depending on the dataset type, a downsampling strategy is applied to limit the number of documents per split (train, validation, test) to make the dataset manageable.
4. **Tokenization**: The raw text documents and their reference summaries are tokenized using a predefined transformer tokenizer.
5. **Entity Recognition (NER)**: Important entities (people, places, organizations) are identified across the text using fine-tuned models. This is a crucial step for establishing connections between documents later on.
6. **Sentence Embedding**: The system computes and stores embeddings representing the semantic meaning of each individual sentence.
7. **Metric Calculation**: Detailed statistics regarding the dataset's characteristics—such as redundancy levels, average summary lengths, and entity density—are computed.
8. **Artifact Caching & Persistence**: All processed data (tokens, entities, embeddings, metrics) are cached to disk. Additionally, the base pre-trained model and tokenizer are saved locally for subsequent phases to utilize without re-downloading.

---

## Phase 2: Core Encoder (`core_encoder/`)
In this phase, the tokenized text is passed through the primary encoder model (a Longformer-Encoder-Decoder variant) to create rich, contextualized representations.

1. **Context Extraction**: Sentences previously grouped into thematic clusters are unpacked into contiguous blocks of text.
2. **Batch Formation**: The text blocks are batched together carefully to optimize processing without overwhelming system resources.
3. **Global Attention Assignment**: A specific global attention mask is constructed. Typically, this mechanism ensures that the model can maintain long-range reasoning over very long documents by forcing specific tokens to attend globally.
4. **Encoder Forward Pass**: The text batches are fed into the encoder model to generate contextualized "hidden states"—the model's internal understanding of the text.
5. **Sharding and Output Storage**: Instead of keeping all generated hidden states in active memory, the pipeline securely shards and writes each document's outputs (hidden states, attention masks, and input sequences) directly to disk.

---

## Phase 3: Cross-Document Fusion (`cross_document_fusion/`)
This is the intellectual core of the pipeline, where the system connects concepts across multiple documents in the same cluster.

1. **Data Loading**: The system streams the previously sharded encoder outputs back into the pipeline cluster by cluster.
2. **Sentence Recovery**: It recovers distinct, sentence-level boundaries from the continuous stream of encoded text, retaining document identification for each sentence.
3. **Cross-Document Interaction**: Sentences from different source documents are allowed to "interact" with one another through a fusion layer. A document-aware bias ensures the model knows whether two sentences came from the same source or different sources.
4. **Entity Alignment**: Entities extracted in Phase 1 are mapped against the fused sentences. This groups information around specific actors or concepts across the entire cluster.
5. **Contradiction Detection**: By leveraging the aligned entities and the original text, the pipeline scans for conflicting claims (e.g., Document A says an event happened on Monday, Document B says Tuesday) and generates contradiction signals.
6. **Contradiction-Aware Fusion**: The system refines its semantic understanding by factoring in the detected contradictions and the aligned entities.
7. **Memory Construction**: A unified "Cross-Document Fusion Memory" is built and cached, containing the highly interconnected semantic state, the mapped entities, and the contradiction markers.

---

## Phase 4: Decoding State Preparation (`decoder/`)
This phase acts as a bridge between the heavily engineered fusion memory and the standard generative decoder model.

1. **Memory Retrieval**: The specific fused semantic memory for a cluster is loaded from the cache.
2. **State Preparation**: The fused sentence representations are reorganized. They are systematically padded or truncated to ensure they perfectly match the strict sequence length limitations expected by the decoder.
3. **Mask Generation**: A corresponding attention mask is synthesized for the decoder to distinguish between actual information and padded filler.

---

## Phase 5: Summary Generation & Evaluation (`summary_generator/`)
The final step translates the fused semantic state into human-readable text and validates its quality.

1. **Configuration Setup**: Decoding rules are established, dictating length constraints, beam search parameters, and repetition penalties.
2. **Hedging Logic Application**: The system checks the contradiction memory from Phase 3. If conflicts were detected between documents, "hedging" logic is activated, encouraging the generator to use careful language (e.g., "sources conflict on..." or "reportedly").
3. **Text Generation**: The generative model uses the prepared states to autoregressively draft the final summary based on the predefined configuration.
4. **Multi-Metric Evaluation (Optional)**: If reference summaries are provided, the generated text is scored against them using standard metrics (like ROUGE) and assessed for factual consistency.
5. **Final Output Archival**: The resulting text, along with metadata detailing the generation parameters, the number of entities utilized, and the detected contradictions, are saved as JSON files and plain text documents in the final outputs directory.
