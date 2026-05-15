import os
import nltk
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables (for MODELS_DIR if set)
load_dotenv()

def download_all():
    print("🚀 Starting comprehensive model download...")

    # 1. NLTK Resources
    print("\n📦 Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("✅ NLTK resources ready.")

    # 2. SpaCy Models
    print("\n📦 Downloading SpaCy models...")
    for model in ["en_core_web_sm", "en_core_web_md"]:
        try:
            if not spacy.util.is_package(model):
                print(f"Downloading {model}...")
                spacy.cli.download(model)
            else:
                print(f"✅ {model} already installed.")
        except Exception as e:
            print(f"⚠️ Failed to download {model}: {e}")

    # 3. HuggingFace Models
    hf_models = [
        "allenai/led-large-16384",
        "cross-encoder/nli-deberta-v3-small",
        "all-MiniLM-L6-v2",
        "facebook/bart-large-mnli"
    ]
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n⚠️ Warning: HF_TOKEN not found in .env file. Downloading as anonymous user.")
    
    print("\n📦 Pre-downloading HuggingFace models...")
    for model_id in hf_models:
        try:
            print(f"Downloading {model_id}...")
            snapshot_download(
                repo_id=model_id, 
                token=hf_token
            )
            print(f"✅ {model_id} cached.")
        except Exception as e:
            print(f"⚠️ Failed to download {model_id}: {e}")

    print("\n✨ All models downloaded and cached successfully!")

if __name__ == "__main__":
    
    download_all()
