from agent1 import sentence_splitting
from sentence_transformers import SentenceTransformer
import numpy as np


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
if device == "cuda":
    print(f"🚀 SentenceTransformer is using GPU ({torch.cuda.get_device_name(0)})")
else:
    print("⚠️ SentenceTransformer is using CPU.")

def generate_sentence_embedding(sentences:list[str])->np.ndarray:
    embeddings = model.encode(sentences)
    return embeddings

def get_embedding(sentence:str)->np.ndarray:
    embedding = model.encode([sentence])
    return embedding[0]
