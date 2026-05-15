import os
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from dotenv import load_dotenv
from utils import (
    check_dataset, analyze, save_cache, setup_cache_dirs,
    save_fine_tuned_transformer, save_fine_tuned_ner
)

# Load environment variables
load_dotenv()

def main():
    # 0. Download required NLTK resources
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"⚠️ Warning: Error downloading nltk resources: {e}")

    # 1. Check if datasets are present
    datasets = check_dataset()
    if not datasets:
        print("❌ No datasets found. Please ensure datasets are saved in the '../datasets/' directory.")
        return

    # 2. Set up the cache directories
    dataset_names = [name.lower().replace('/', '_') for name in datasets.keys()]
    setup_cache_dirs(dataset_names)

    # 3. Define sampling targets - Pull from .env with defaults
    train_size = int(os.getenv("TRAIN_SAMPLE_SIZE", 100))
    val_size = int(os.getenv("VAL_SAMPLE_SIZE", 50))
    test_size = int(os.getenv("TEST_SAMPLE_SIZE", 100))

    sampling_targets = {
        'CNN/Daily-News': {'train': train_size, 'validation': val_size, 'test': test_size},
        'Multi-News': {'train': train_size, 'validation': val_size, 'test': test_size}
    }

    splits = ['train', 'validation', 'test']

    # 4. Process each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*20} Analyzing Dataset: {dataset_name} {'='*20}")
        
        for split in tqdm(splits, desc=f"Splits in {dataset_name}"):
            if split not in dataset:
                print(f"⚠️ Split '{split}' not found in {dataset_name}. Skipping...")
                continue
                
            print(f"\n--- Processing Split: {split} ---")
            split_dataset = dataset[split]
            
            # Determine column names based on dataset
            article_col = 'article'
            summary_col = 'highlights'
            if dataset_name == 'Multi-News':
                article_col = 'document'
                summary_col = 'summary'
                
            # Filter out corrupt/empty records before sampling
            initial_len = len(split_dataset)
            split_dataset = split_dataset.filter(
                lambda x: x[article_col] is not None and str(x[article_col]).strip() != "" and 
                          x[summary_col] is not None and str(x[summary_col]).strip() != ""
            )
            filtered_len = len(split_dataset)
            if initial_len != filtered_len:
                print(f"⚠️ Warning: Filtered out {initial_len - filtered_len} empty/corrupted records.")
                
            # 5. Apply Downsampling
            target_size = sampling_targets.get(dataset_name, {}).get(split)
            if target_size and len(split_dataset) > target_size:
                print(f"Sampling {target_size} documents from {len(split_dataset)} total...")
                process_dataset = split_dataset.select(range(target_size))
            else:
                process_dataset = split_dataset
                
            # 6. Analyze the dataset
            try:
                analysis_output = analyze(
                    process_dataset, 
                    article_col=article_col, 
                    summary_col=summary_col, 
                    dataset_name=dataset_name
                )
                
                # 7. Save analysis results to cache
                cache_name = f"{dataset_name}_{split}"
                save_cache(
                    dataset_name=cache_name, 
                    tokenized_data=analysis_output['tokenized'],
                    entities=analysis_output['entities'],
                    embeddings=analysis_output['embeddings'],
                    analysis_results=analysis_output['metrics']
                )
                
                # Display summary results
                print(f"\nResults for {dataset_name} ({split}):")
                for key, value in analysis_output['metrics'].items():
                    if not isinstance(value, (pd.Series, list, np.ndarray)):
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"❌ Error analyzing {dataset_name} ({split}): {e}")
                
        print(f"\n{'='*60}\n")
        
        # 8. Model Persistence
        from transformers import AutoModelForSeq2SeqLM
        from utils import get_tokenizer
        
        try:
            print(f"📦 Persisting model checkpoint for {dataset_name}...")
            tokenizer = get_tokenizer("allenai/led-large-16384")
            model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384")
            
            # Use environment variable for models directory
            model_output_dir = os.getenv("MODELS_DIR", "models")
            model_save_name = "final_mds_led"
            
            save_fine_tuned_transformer(model, tokenizer, output_dir=model_output_dir, model_name=model_save_name)
        except Exception as e:
            print(f"⚠️ Model persistence failed: {e}")

if __name__ == "__main__":
    main()

