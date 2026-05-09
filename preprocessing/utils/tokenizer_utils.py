from transformers import AutoTokenizer

_tokenizer_cache = {}

def get_tokenizer(model_name: str = "allenai/led-large-16384"):
    if model_name not in _tokenizer_cache:
        try:
            _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print("Error: Tokenizer Load unsucessful: ", e)
            exit(1)
    return _tokenizer_cache[model_name]