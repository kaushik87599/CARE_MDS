import spacy
import spacy.cli
import os
import sys

_nlp = None

def _get_nlp():
    """
    Lazy-loads the SpaCy model and handles GPU configuration and model downloading.
    """
    global _nlp
    if _nlp is None:
        model_name = 'en_core_web_md'
        try:
            # Enable GPU if available
            gpu_available = spacy.prefer_gpu()
            if gpu_available:
                print("🚀 SpaCy is using GPU acceleration!")
            else:
                print("⚠️ SpaCy is using CPU (GPU not found or not configured).")
            
            try:
                _nlp = spacy.load(model_name)
            except (OSError, ImportError):
                print(f"⚠️ SpaCy model '{model_name}' not found. Downloading...")
                spacy.cli.download(model_name)
                try:
                    # Attempt direct import to avoid needing a runtime restart in some environments (like Colab)
                    import en_core_web_md
                    _nlp = en_core_web_md.load()
                except ImportError:
                    _nlp = spacy.load(model_name)
                print(f"✅ Successfully downloaded and loaded '{model_name}'.")
                
        except Exception as e:
            print(f"❌ Error: SpaCy NER Load unsuccessful for '{model_name}': {e}")
            # Instead of exit(1), we'll let the calling functions handle the None case or fallback
            # but for consistency with existing code, keeping a print.
            return None
    return _nlp

def count_entities(text):
    """
    Counts the number of entities in a single text string.
    """
    nlp = _get_nlp()
    if nlp is None: return 0
    try:
        doc = nlp(text)
        return len(doc.ents)
    except Exception as e:
        print(f"Error: SpaCy NER failed on text snippet: {str(text)[:100]}...")
        return 0

def count_entities_batch(texts, batch_size=64):
    """
    Counts entities in a list of texts using optimized batch processing.
    """
    nlp = _get_nlp()
    if nlp is None: return [0] * len(texts)

    # Optimization: Use n_process for CPU, must be 1 for GPU
    n_process = 1
    # spacy.prefer_gpu() returns True if GPU is already active or was just activated
    if not spacy.prefer_gpu():
        n_process = os.cpu_count() or 1
        
    from tqdm.auto import tqdm
    
    # Components to disable for speed (we only need 'ner' and 'tok2vec'/'transformer')
    disabled_pipes = [p for p in nlp.pipe_names if p not in ["ner", "tok2vec", "transformer"]]
    
    try:
        return [len(doc.ents) for doc in tqdm(
            nlp.pipe(texts, batch_size=batch_size, n_process=n_process, disable=disabled_pipes), 
            total=len(texts), 
            desc="NER Entity Counting",
            leave=False
        )]
    except Exception as e:
        print(f"Error in batch processing: {e}")
        # Fallback to individual processing if batch fails
        return [count_entities(t) for t in texts]

def extract_entities(text):
    """
    Extracts entities from text as a list of (text, label) tuples.
    """
    nlp = _get_nlp()
    if nlp is None: return []
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        print(f"Error: SpaCy NER failed on text snippet: {str(text)[:100]}...")
        return []

def extract_entities_batch(texts, batch_size=64):
    """
    Extracts entities from a list of texts in batches.
    Returns a list of lists of (text, label) tuples.
    """
    nlp = _get_nlp()
    if nlp is None: return [[] for _ in texts]
    
    n_process = 1
    if not spacy.prefer_gpu():
        n_process = os.cpu_count() or 1
        
    from tqdm.auto import tqdm
    
    # Components to disable for speed
    disabled_pipes = [p for p in nlp.pipe_names if p not in ["ner", "tok2vec", "transformer"]]
    
    results = []
    try:
        for doc in tqdm(
            nlp.pipe(texts, batch_size=batch_size, n_process=n_process, disable=disabled_pipes), 
            total=len(texts), 
            desc="NER Entity Extraction",
            leave=False
        ):
            results.append([(ent.text, ent.label_) for ent in doc.ents])
    except Exception as e:
        print(f"Error in batch entity extraction: {e}")
        # Fallback
        return [extract_entities(t) for t in texts]
        
    return results