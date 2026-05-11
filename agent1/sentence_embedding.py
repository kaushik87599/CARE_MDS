from agent1 import sentence_splitting
from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_sentence_embedding(sentences:list[str])->np.ndarray:
    embeddings = model.encode(sentences)
    return embeddings

def get_embedding(sentence:str)->np.ndarray:
    embedding = model.encode([sentence])
    return embedding[0]
