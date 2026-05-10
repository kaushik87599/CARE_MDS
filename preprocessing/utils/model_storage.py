import os
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def save_fine_tuned_transformer(model, tokenizer, output_dir, model_name="fine_tuned_model"):
    """
    Saves the fine-tuned transformer model and tokenizer.
    Follows the rule: Only save if fine-tuned, otherwise reload from HuggingFace to save space.
    """
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"💾 Saving fine-tuned transformer to {save_path}...")
    
    # save_pretrained is efficient (uses safetensors/bin)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"✅ Transformer model and tokenizer saved to {save_path}")

def save_fine_tuned_ner(nlp, output_dir, model_name="fine_tuned_ner"):
    """
    Saves the fine-tuned SpaCy NER model.
    """
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"💾 Saving fine-tuned NER to {save_path}...")
    
    # SpaCy's to_disk is the standard way to save models
    nlp.to_disk(save_path)
    
    print(f"✅ NER model saved to {save_path}")

def save_model_weights_only(model, output_dir, filename="weights.pt"):
    """
    Saves only the state dict (weights) of a model using torch.save for efficiency.
    Useful for custom architectures or when you want to avoid saving full model objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    
    print(f"💾 Saving model weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model weights saved to {save_path}")

if __name__ == "__main__":
    # Example usage (placeholders)
    print("Model Storage Utility Ready.")
    print("Note: Always prefer reloading from HuggingFace for base models to save space.")
