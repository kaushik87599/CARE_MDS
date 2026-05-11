from pandas._libs.tslibs import timedeltas
from pandas._libs import indexing
import sys
import os
import nltk
from utils import load_multi_dataset

nltk.download("punkt")


import pandas as pd
from preprocessing import check_dataset

def document_split(text: str) -> list[str]:
    documents = text.split("|||||")
    return documents


def sentence_split(document:str)->list[str]:
    
    try:
        from nltk.tokenize import sent_tokenize
        import nltk
        nltk.data.find('tokenizers/punkt')
    except (ImportError, LookupError) as e:
        print(f"Error: {e}")
        if isinstance(e, ImportError):
            print("Error: The 'nltk' library is not installed.")
            print('Trying to install nltk...')
            try:
                os.system('pip install nltk')
            except Exception as e:
                print(f"Error: Failed to install nltk: {e}")
                print("Fix: Run 'pip install nltk' in your terminal.")
                exit(1)
        elif isinstance(e, LookupError):
            print("Error: The 'punkt' tokenizer model is missing.")
            print("Fix: Run 'nltk.download(\"punkt\")' in your Python console.")
            try:
                print('Trying to download nltk punkt...')
                nltk.download("punkt")
            except Exception as e:
                print(f"Error: Failed to download punkt: {e}")
                print("Fix: Run 'nltk.download(\"punkt\")' in your Python console.")
                exit(1)
        exit(1)
    
    sentences = []
    sentences.extend(sent_tokenize(document))
    return sentences
    
    
if __name__ == "__main__":
    df = load_multi_dataset()
    
