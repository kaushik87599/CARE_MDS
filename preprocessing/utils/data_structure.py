import pandas as pd
import re 
import torch
import numpy as np
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

try:
    from .tokenizer_utils import get_tokenizer
    from .ner_utils import count_entities, count_entities_batch, extract_entities_batch
except ImportError:
    from tokenizer_utils import get_tokenizer
    from ner_utils import count_entities, count_entities_batch, extract_entities_batch

# Global model caches to avoid reloading
_embed_model = None
_nli_pipe = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        model_id = 'all-MiniLM-L6-v2'
        try:
            print(f"🔄 Loading embedding model: {model_id}...")
            _embed_model = SentenceTransformer(model_id)
            print(f"✅ Embedding model '{model_id}' is ready.")
        except Exception as e:
            print(f"⚠️ Initial load of '{model_id}' failed: {e}")
            print(f"🚀 Attempting to download/re-verify '{model_id}'...")
            try:
                _embed_model = SentenceTransformer(model_id)
                print(f"✅ Successfully initialized '{model_id}' after retry.")
            except Exception as retry_error:
                print(f"❌ CRITICAL: Could not load embedding model '{model_id}'. Error: {retry_error}")
                raise retry_error
    return _embed_model

def get_nli_pipe():
    global _nli_pipe
    if _nli_pipe is None:
        model_id = "cross-encoder/nli-deberta-v3-small"
        try:
            print(f"🔄 Initializing NLI pipeline with '{model_id}'...")
            device = 0 if torch.cuda.is_available() else -1
            _nli_pipe = pipeline("text-classification", model=model_id, device=device)
            print(f"✅ NLI pipeline '{model_id}' is ready.")
        except Exception as e:
            print(f"⚠️ Initial load of NLI model '{model_id}' failed: {e}")
            print(f"🚀 Attempting to download/re-verify '{model_id}'...")
            try:
                device = 0 if torch.cuda.is_available() else -1
                _nli_pipe = pipeline("text-classification", model=model_id, device=device)
                print(f"✅ Successfully initialized NLI model '{model_id}' after retry.")
            except Exception as retry_error:
                print(f"❌ CRITICAL: Could not load NLI model '{model_id}'. Error: {retry_error}")
                raise retry_error
    return _nli_pipe

def analyze(dataset, article_col='article', summary_col='highlights', dataset_name=None)->dict:
    """
    Analyzes the dataset and prepares data for caching.
    Calculates:
    1. Number of documents
    2. Average token length (articles & summaries)
    3. Redundancy level
    4. Contradiction frequency
    5. Entity density
    Returns a dictionary with both summary metrics and raw data for caching.
    """
    start_time = time.time()
    results = {}
    metrics = {}
    tokenizer = get_tokenizer()
    
    articles = list(dataset[article_col])
    summaries = list(dataset[summary_col])
    num_docs = len(articles)
    
    metrics['num_docs'] = num_docs
    print(f"Analyzing {num_docs} documents...")

    # 1. Tokenization & Lengths
    t0 = time.time()
    tokenized_articles = tokenizer(articles, truncation=True, max_length=4096, padding=False, add_special_tokens=True)
    tokenized_summaries = tokenizer(summaries, truncation=True, max_length=1024, padding=False, add_special_tokens=True)
    
    article_lengths = [len(x) for x in tokenized_articles["input_ids"]]
    summary_lengths = [len(x) for x in tokenized_summaries["input_ids"]]
    
    metrics['avg_article_token_len'] = float(np.mean(article_lengths))
    metrics['avg_summary_token_len'] = float(np.mean(summary_lengths))
    print(f'Finished tokenization in {time.time()-t0:.2f}s')

    # 2. Sentence Embeddings (For caching and redundancy)
    t0 = time.time()
    embed_model = get_embed_model()
    # Process in batches for memory efficiency
    embeddings = embed_model.encode(articles, batch_size=32, show_progress_bar=True)
    print(f'Finished sentence embeddings in {time.time()-t0:.2f}s')

    # 3. Redundancy Analysis (Sampled for speed, but using full embeddings where possible)
    def calculate_redundancy_from_embeddings(idx, text):
        embed_model = get_embed_model()
        if '|||||' in text:
            # Multi-document case: split and encode sub-docs
            docs = [d.strip() for d in text.split('|||||') if len(d.strip()) > 20]
            if len(docs) < 2: return 0.0
            sub_embeddings = embed_model.encode(docs, show_progress_bar=False)
            sim_matrix = cosine_similarity(sub_embeddings)
        else:
            # Single document case: use sentence tokenization
            import nltk
            docs = nltk.sent_tokenize(text)
            if len(docs) < 2: return 0.0
            sub_embeddings = embed_model.encode(docs, show_progress_bar=False)
            sim_matrix = cosine_similarity(sub_embeddings)
        
        mask = np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)
        redundant_pairs = sim_matrix[mask] > 0.85
        return float(np.mean(redundant_pairs)) if len(redundant_pairs) > 0 else 0.0

    sample_size = min(100, num_docs)
    redundancy_scores = [calculate_redundancy_from_embeddings(i, articles[i]) for i in tqdm(range(sample_size), desc="Redundancy", leave=False)]
    metrics['redundancy_level'] = float(np.mean(redundancy_scores))

    # 4. Contradiction Frequency (NLI)
    t0 = time.time()
    nli_pipe = get_nli_pipe()
    
    def calculate_contradiction(text):
        if '|||||' in text:
            docs = [d.strip()[:512] for d in text.split('|||||') if len(d.strip()) > 50]
            if len(docs) < 2: return 0.0
            # Check first pair for sample
            result = nli_pipe(f"{docs[0]} [SEP] {docs[1]}")
            label = result[0]['label'].lower()
            return 1.0 if 'contradiction' in label else 0.0
        return 0.0

    contradiction_scores = [calculate_contradiction(art) for art in tqdm(articles[:min(50, num_docs)], desc="Contradiction", leave=False)]
    metrics['contradiction_frequency'] = float(np.mean(contradiction_scores))
    print(f'Finished NLI analysis in {time.time()-t0:.2f}s')

    # 5. Entity Density & Extraction
    t0 = time.time()
    batch_size = 32 if torch.cuda.is_available() else 128
    # Extract actual entities for reuse in EntityAligner
    all_entities = extract_entities_batch(articles, batch_size=batch_size)
    entity_counts = [len(ents) for ents in all_entities]
    
    # Calculate density per article
    densities = [count / length if length > 0 else 0 for count, length in zip(entity_counts, article_lengths)]
    metrics['avg_entity_density'] = float(np.mean(densities))
    metrics['entity_density_list'] = densities # Store list for analysis_results.json
    print(f'Finished entity metrics & extraction in {time.time()-t0:.2f}s')

    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

    # Prepare final output
    results['metrics'] = metrics
    results['tokenized'] = tokenized_articles
    results['entities'] = all_entities # Store actual entity strings (list of lists of (text, label) tuples)
    results['embeddings'] = embeddings

    print(f"Total analysis time: {time.time() - start_time:.2f}s")
    return results
    

    