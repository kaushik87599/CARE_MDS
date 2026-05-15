from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration
)

import torch
import sys
import pickle
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Centralized Model Path
def get_model_path():
    models_dir = os.getenv("MODELS_DIR", "models")
    return os.path.join(models_dir, "final_mds_led")

def load_saved_tokenizer(path=None):
    if path is None:
        path = get_model_path()
    set_device()
    print(f"Loading tokenizer from {path}...")
    try:
        if not os.path.exists(path):
            print(f"ERROR: Model path '{path}' does not exist.")
            sys.exit(1)
        tokenizer = LEDTokenizer.from_pretrained(path)
        print('Successfully loaded tokenizer.')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load tokenizer from {path}.")
        print(f"Details: {e}")
        sys.exit(1)
    return tokenizer

def load_saved_model(path=None):
    if path is None:
        path = get_model_path()
    device = set_device()
    print(f"Loading model to {device} from {path}...")
    try:
        if not os.path.exists(path):
            print(f"ERROR: Model path '{path}' does not exist.")
            sys.exit(1)
            
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        model = LEDForConditionalGeneration.from_pretrained(
            path, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        print(f"📦 Moving model to {device} memory...")
        model.to(device)
        print(f'✅ Successfully loaded model.')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model from {path}.")
        print(f"Details: {e}")
        sys.exit(1)
    
    return model

def get_encoder_from_model():
    model = load_saved_model()
    encoder = model.get_encoder()
    return encoder

def load_packed_context():
    packed_cache_dir = os.getenv("PACKED_CACHE_DIR", "cache/cache")
    file_path = os.path.join(packed_cache_dir, "packed_contexts.pkl")
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: Packed contexts file '{file_path}' not found.")
            sys.exit(1)
        with open(file_path, "rb") as f:
            packed_contexts = pickle.load(f)
        print(f"Successfully loaded packed contexts from {file_path}.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load packed contexts from {file_path}.")
        print(f"Details: {e}")
        sys.exit(1)
   
    return packed_contexts

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device.type.upper()} for computation")
    return device
    