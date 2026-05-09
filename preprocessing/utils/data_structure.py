import pandas as pd
import re 

try:
    from .tokenizer_utils import get_tokenizer
    from .ner_utils import count_entities, count_entities_batch
except ImportError:
    from tokenizer_utils import get_tokenizer
    from ner_utils import count_entities, count_entities_batch
 

def analyze(df:pd.DataFrame)->dict:
    structure = dict()
    # structure tell the following properties of the dataset :
    # 1. Number of documents
    # 2. tokenizer used
    # 3. Average token length of article
    # 4. Summary length
    # 5. Redundancy level
    # 6.  Contradiction frequency
    # 7.  Entity density
    
    
    tokenizer = get_tokenizer()
    print('Finished loading tokenizer')
    structure['len'] = (len(df))
    print('Finished calcualting length of dataset')
    structure['tokenizer'] = 'allenai/led-large-16384'
    print('Finished storing tokenizer')

    def get_token_lengths(texts, batch_size=64):
        lengths = []
        texts_list = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i:i+batch_size]
            tokens = tokenizer(batch, truncation=False, padding=False)
            lengths.extend([len(x) for x in tokens["input_ids"]])
        return lengths

    structure['avg_article_token_len'] = pd.Series(get_token_lengths(df['article'])).mean()
    print('Finished calcualting average article token length')
    structure['avg_summary_token_len'] = pd.Series(get_token_lengths(df['highlights'])).mean()
    print('Finished calcualting average summary token length')
    structure["entity_count"] = count_entities_batch(df["article"].tolist(), batch_size=64)
    print('Finished calcualting entity count')
    structure["entity_density"] = structure["entity_count"] / structure['avg_article_token_len']
    print('Finished calcualting entity density')
    print(f"Average entity density: {structure['entity_density'].mean():.4f}")   


    
    return structure
    

    