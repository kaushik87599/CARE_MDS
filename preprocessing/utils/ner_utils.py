import spacy

_nlp = None

def count_entities(text):
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_trf')
        except Exception as e:
            print("Error: SpaCy NER Load unsucessful: ", e)
            exit(1)

    try:
        doc = _nlp(text)
        entities = doc.ents
    except Exception as e:
        print("Error: SpaCy NER failed on: ", text)
        return 0
    return len(entities)

def count_entities_batch(texts, batch_size=64):
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load('en_core_web_trf')
        except Exception as e:
            print("Error: SpaCy NER Load unsucessful: ", e)
            exit(1)

    # nlp.pipe is heavily optimized for processing multiple texts at once
    return [len(doc.ents) for doc in _nlp.pipe(texts, batch_size=batch_size)]