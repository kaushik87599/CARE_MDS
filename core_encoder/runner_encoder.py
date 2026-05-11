import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
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

def save_encoder_output(cluster_id, hidden_states, input_ids, attention_mask):
    """
    STEP 10 — Save Encoder Outputs
    """
    save_dir = "cache/encoder_outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    save_data = {
        "cluster_id": cluster_id,
        "hidden_states": hidden_states.cpu(),
        "attention_mask": attention_mask.cpu(),
        "input_ids": input_ids.cpu()
    }
    
    file_path = os.path.join(save_dir, f"{cluster_id}.pt")
    torch.save(save_data, file_path)

class ClusterDataset(Dataset):
    def __init__(self, cluster_texts):
        self.data = cluster_texts
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

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

        # Batching Configuration
        BATCH_SIZE = 4 # Conservative for 15GB VRAM. Adjust if needed.
        dataset = ClusterDataset(cluster_texts)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Processing {len(cluster_texts)} clusters with batch size {BATCH_SIZE}...")

        for batch in tqdm(dataloader, desc="Encoding batches"):
            cluster_ids = batch["cluster_id"]
            texts = batch["text"]

            # Tokenize Batch
            inputs = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=4096,
                return_tensors="pt"
            ).to(device)

            # Create Global Attention Mask
            global_attention_mask = create_global_attention_mask(inputs["input_ids"])

            # Run Encoder Forward Pass
            outputs = run_encoder_forward_pass(encoder, inputs, global_attention_mask)

            # Extract Hidden States
            batch_hidden_states = extract_hidden_states(outputs)
            
            # SECTION 7 — Validate Dimensions
            if batch_hidden_states.ndim != 3:
                print(f"CRITICAL ERROR: Hidden states dimension mismatch. Expected 3 (batch, seq, dim)")
                sys.exit(1)
            
            if batch_hidden_states.shape[-1] != 1024:
                print(f"CRITICAL ERROR: Hidden dimension mismatch. Expected 1024 (LED-large), got {batch_hidden_states.shape[-1]}")
                sys.exit(1)

            # STEP 10: Save Encoder Outputs Individually
            for i in range(len(cluster_ids)):
                save_encoder_output(
                    cluster_ids[i], 
                    batch_hidden_states[i:i+1], # Slice to keep 3D shape
                    inputs["input_ids"][i:i+1],
                    inputs["attention_mask"][i:i+1]
                )
            
            # CLEAR GPU CACHE is generally not needed every batch with FP16 + Batching 
            # unless we are right at the edge of OOM.
        
        # FINAL CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Saved all encoder outputs.")
        print("Core encoder pipeline completed successfully.")
        return None # Return None as we no longer store results in memory

    except torch.cuda.OutOfMemoryError:
        print("CRITICAL ERROR: GPU Out of Memory. Consider reducing max_length or using a smaller batch size.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred during the encoder forward pass.")
        print(f"Details: {e}")
        sys.exit(1)


    


if __name__ == "__main__":
    run_core_encoder()

    