from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration
)

import torch
import sys
import pickle
import os



MODEL_PATH = "models/models/final_mds_led"

def load_saved_tokenizer(path = MODEL_PATH):
    set_device()
    print("Loading tokenizer...")
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

def load_saved_model(path = MODEL_PATH):
    device = set_device()
    print(f"Loading model to {device}...")
    try:
        if not os.path.exists(path):
            print(f"ERROR: Model path '{path}' does not exist.")
            sys.exit(1)
            
        # Optimization: Use float16 if on GPU
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        model = LEDForConditionalGeneration.from_pretrained(
            path, 
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        model.to(device)
        print(f'Successfully loaded model in {dtype}.')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model from {path}.")
        print(f"Details: {e}")
        sys.exit(1)
    
    return model

def get_encoder_from_model():
    """
    returns the encoder part of the model 
    """
    model = load_saved_model()
    encoder = model.get_encoder()
    return encoder

def get_decoder_from_model():
    """
    returns the decoder part of the model 
    """
    model = load_saved_model()
    decoder = model.get_decoder()
    return decoder

def load_packed_context():
    '''
    use pickle to load the packed contexts.
    
    '''
    file_path = "cache/cache/packed_contexts.pkl"
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: Packed contexts file '{file_path}' not found. Ensure Phase 4 is completed.")
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
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
    )
    if(device.type == "cuda"):
        print("Using GPU for computation")
    else:
        print("Using CPU for computation")
    return device
    