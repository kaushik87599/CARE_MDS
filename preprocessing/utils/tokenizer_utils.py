from transformers import AutoTokenizer

_tokenizer_cache = {}

def get_tokenizer(model_name: str = "allenai/led-large-16384"):
    if model_name not in _tokenizer_cache:
        try:
            print(f"🔄 Loading tokenizer: {model_name}...")
            _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ Tokenizer '{model_name}' is ready.")
        except Exception as e:
            print(f"⚠️ Error: Tokenizer Load unsuccessful for '{model_name}': {e}")
            print(f"🚀 Attempting to download/verify '{model_name}'...")
            try:
                _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
                print(f"✅ Successfully initialized '{model_name}' after retry.")
            except Exception as retry_error:
                print(f"❌ CRITICAL: Could not load tokenizer '{model_name}'. Error: {retry_error}")
                exit(1)
    return _tokenizer_cache[model_name]