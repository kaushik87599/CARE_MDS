import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
from .loader import (load_packed_context, load_saved_tokenizer, get_encoder_from_model, set_device)

# Load environment variables
load_dotenv()

def extract_sentences(packed_contexts):
    """
    Convert packed contexts into text blocks.
    """
    cluster_texts = []
    print("Converting packed contexts into text blocks...")
    for cluster in tqdm(packed_contexts, desc="Extracting text"):
        try:
            if "cluster_id" not in cluster or "packed_context" not in cluster:
                continue
            cluster_id = cluster["cluster_id"]
            sentences = [item[0] for item in cluster["packed_context"]]
            if not sentences:
                continue
            text_block = " ".join(sentences)
            cluster_texts.append({"cluster_id": cluster_id, "text": text_block})
        except Exception:
            continue
    print(f"Successfully converted {len(cluster_texts)} clusters.")
    return cluster_texts

def create_global_attention_mask(input_ids):
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1
    return global_attention_mask

def run_encoder_forward_pass(encoder, inputs, global_attention_mask):
    with torch.inference_mode():
        outputs = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=global_attention_mask
        )
    return outputs

def extract_hidden_states(outputs):
    return outputs.last_hidden_state

def save_encoder_output(cluster_id, hidden_states, input_ids, attention_mask):
    """
    Saves individual encoder outputs.
    """
    save_dir = os.getenv("ENCODER_OUT_DIR", "cache/encoder_outputs")
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
        
        BATCH_SIZE = 4 
        dataset = ClusterDataset(cluster_texts)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"🚀 Encoding {len(cluster_texts)} clusters...")

        for batch in tqdm(dataloader, desc="Encoding batches"):
            try:
                cluster_ids = batch["cluster_id"]
                texts = batch["text"]

                inputs = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=4096,
                    return_tensors="pt"
                ).to(device)

                global_attention_mask = create_global_attention_mask(inputs["input_ids"])
                outputs = run_encoder_forward_pass(encoder, inputs, global_attention_mask)
                batch_hidden_states = extract_hidden_states(outputs)
                
                if batch_hidden_states.ndim != 3 or batch_hidden_states.shape[-1] != 1024:
                    continue

                for i in range(len(cluster_ids)):
                    save_encoder_output(
                        cluster_ids[i].item() if isinstance(cluster_ids[i], torch.Tensor) else cluster_ids[i], 
                        batch_hidden_states[i:i+1],
                        inputs["input_ids"][i:i+1],
                        inputs["attention_mask"][i:i+1]
                    )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception:
                continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Core encoder pipeline completed successfully.")
        return None 

    except Exception as e:
        print(f"CRITICAL ERROR in encoder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_core_encoder()


    