import torch
import os
import sys
from tqdm import tqdm
from loader import (load_packed_context, load_saved_tokenizer, get_encoder_from_model, set_device)

def extract_sentences(packed_contexts):
    """
    Convert packed contexts into text blocks.
    
    Args:
        packed_contexts: List of clusters, where each cluster has a list of (sentence, score, embedding, doc_id) tuples.
    
    Returns:
        List of dictionaries with 'cluster_id' and 'text' (concatenated sentences).
    """
    cluster_texts = []
    print("Converting packed contexts into text blocks...")
    
    # Iterate through each cluster
    for cluster in tqdm(packed_contexts, desc="Extracting text"):
        if "cluster_id" not in cluster or "packed_context" not in cluster:
            print(f"CRITICAL ERROR: Invalid cluster structure in packed contexts. Keys 'cluster_id' and 'packed_context' are required.")
            sys.exit(1)
            
        cluster_id = cluster["cluster_id"]
        
        # Extract only the sentence string (the 0th element in each tuple)
        try:
            sentences = [item[0] for item in cluster["packed_context"]]
        except (IndexError, TypeError) as e:
            print(f"CRITICAL ERROR: Invalid packed sentence format in cluster {cluster_id}. Expected (sentence, score, embedding, doc_id).")
            print(f"Details: {e}")
            sys.exit(1)
        
        # Convert into one text block
        text_block = " ".join(sentences)
        
        cluster_texts.append({
            "cluster_id": cluster_id,
            "text": text_block
        })
    
    print(f"Successfully converted {len(cluster_texts)} clusters into text blocks.")
    return cluster_texts

def create_global_attention_mask(input_ids):
    """
    Create Global Attention Mask
    By default, sets the first token to have global attention.
    """
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1
    return global_attention_mask

def run_encoder_forward_pass(encoder, inputs, global_attention_mask):
    """
    Run Encoder Forward Pass
    """
    with torch.inference_mode():
        outputs = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=global_attention_mask
        )
    return outputs

def extract_hidden_states(outputs):
    """
    Extract Hidden States
    """
    hidden_states = outputs.last_hidden_state
    # (batch_size, seq_len, hidden_dim)
    return hidden_states

def save_encoder_output(cluster_id, hidden_states, inputs):
    """
    STEP 10 — Save Encoder Outputs
    """
    save_dir = "cache/encoder_outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    save_data = {
        "cluster_id": cluster_id,
        "hidden_states": hidden_states.cpu(),
        "attention_mask": inputs["attention_mask"].cpu(),
        "input_ids": inputs["input_ids"].cpu()
    }
    
    file_path = os.path.join(save_dir, f"{cluster_id}.pt")
    torch.save(save_data, file_path)
    # print(f"Saved encoder output for cluster {cluster_id} to {file_path}")

def run_core_encoder():
    print("Starting core encoder...")

    try:
        device = set_device()
        tokenizer = load_saved_tokenizer()
        encoder = get_encoder_from_model()
        encoder.to(device)
        encoder.eval()
        
        packed_contexts = load_packed_context()
        cluster_texts = extract_sentences(packed_contexts)

        all_hidden_states = []

        for cluster in tqdm(cluster_texts, desc="Encoding clusters"):
            
            # Tokenize Packed Text
            inputs = tokenizer(
                cluster["text"],
                truncation=True,
                padding="max_length",
                max_length=4096,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Create Global Attention Mask
            global_attention_mask = create_global_attention_mask(inputs["input_ids"])

            # Run Encoder Forward Pass
            outputs = run_encoder_forward_pass(encoder, inputs, global_attention_mask)

            # Extract Hidden States
            hidden_states = extract_hidden_states(outputs)
            
            # SECTION 7 — Validate Dimensions
            if hidden_states.ndim != 3:
                print(f"CRITICAL ERROR: Hidden states dimension mismatch. Expected 3 (batch, seq, dim), got {hidden_states.ndim}")
                sys.exit(1)
            
            if hidden_states.shape[-1] != 1024:
                print(f"CRITICAL ERROR: Hidden dimension mismatch. Expected 1024 (LED-large), got {hidden_states.shape[-1]}")
                sys.exit(1)

            # print(f"Extracted hidden states with shape: {hidden_states.shape}")
            
            # STEP 10: Save Encoder Outputs
            save_encoder_output(cluster["cluster_id"], hidden_states, inputs)
            
            all_hidden_states.append({
                "cluster_id": cluster["cluster_id"],
                "hidden_states": hidden_states.cpu() # Store on CPU to save memory
            })

            # CLEAR GPU CACHE (Only if needed, currently offloaded to end for speed)
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        
        # FINAL CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Saved encoder outputs")
        print("Core encoder pipeline completed successfully.")
        return all_hidden_states

    except torch.cuda.OutOfMemoryError:
        print("CRITICAL ERROR: GPU Out of Memory. Consider reducing max_length or using a smaller batch size.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred during the encoder forward pass.")
        print(f"Details: {e}")
        sys.exit(1)


    


if __name__ == "__main__":
    run_core_encoder()

    