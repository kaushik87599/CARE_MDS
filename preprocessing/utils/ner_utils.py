import spacy
import spacy.cli
import os

_nlp = None

def _get_nlp():
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
                    # Attempt direct import to avoid needing a runtime restart
                    import en_core_web_md
                    _nlp = en_core_web_md.load()
                except ImportError:
                    _nlp = spacy.load(model_name)
                print(f"✅ Successfully downloaded and loaded '{model_name}'.")
                
        except Exception as e:
            print(f"❌ Error: SpaCy NER Load unsuccessful for '{model_name}': {e}")
            exit(1)
    return _nlp

def count_entities(text):
    nlp = _get_nlp()
    try:
        doc = nlp(text)
        entities = doc.ents
    except Exception as e:
        print("Error: SpaCy NER failed on: ", text[:100], "...")
        return 0
    return len(entities)

def count_entities_batch(texts, batch_size=64):
    nlp = _get_nlp()

    # Optimization: Use n_process for CPU to speed up extraction
    n_process = 1
    if not spacy.prefer_gpu():
        n_process = os.cpu_count() or 1
        
    # nlp.pipe is heavily optimized for processing multiple texts at once
    return [len(doc.ents) for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process)]

def extract_entities(text):
    """
    Extracts entities from text as a list of (text, label) tuples.
    """
    nlp = _get_nlp()
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        print("Error: SpaCy NER failed on: ", text[:100], "...")
        return []

def extract_entities_batch(texts, batch_size=64):
    """
    Extracts entities from a list of texts in batches.
    Returns a list of lists of (text, label) tuples.
    """
    nlp = _get_nlp()
    n_process = 1
    if not spacy.prefer_gpu():
        n_process = os.cpu_count() or 1
        
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        results.append([(ent.text, ent.label_) for ent in doc.ents])
    return results