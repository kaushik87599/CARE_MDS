import os
import pandas as pd
import numpy as np
import nltk
from utils import (
    check_dataset, analyze, save_cache, setup_cache_dirs,
    save_fine_tuned_transformer, save_fine_tuned_ner
)

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
    # Use clean names for directory creation
    dataset_names = [name.lower().replace('/', '_') for name in datasets.keys()]
    setup_cache_dirs(dataset_names)

    # 3. Define sampling targets according to task requirements
    sampling_targets = {
        'CNN/Daily-News': {'train': 8000, 'validation': 1000, 'test': 1000},
        'Multi-News': {'train': 4000, 'validation': 500, 'test': 500}
    }

    splits = ['train', 'validation', 'test']

    # 4. Process each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*20} Analyzing Dataset: {dataset_name} {'='*20}")
        
        for split in splits:
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
                
            # 5. Apply Downsampling
            target_size = sampling_targets.get(dataset_name, {}).get(split)
            if target_size and len(split_dataset) > target_size:
                print(f"Sampling {target_size} documents from {len(split_dataset)} total...")
                process_dataset = split_dataset.select(range(target_size))
            else:
                process_dataset = split_dataset
                
            # 6. Analyze the dataset (Uses ner_utils and tokenizer_utils internally)
            try:
                analysis_output = analyze(
                    process_dataset, 
                    article_col=article_col, 
                    summary_col=summary_col, 
                    dataset_name=dataset_name
                )
                
                # 7. Save analysis results to cache
                # We include the split in the name to ensure save_cache can organize them
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
                import traceback
                traceback.print_exc()
                
        print(f"\n{'='*60}\n")
        
        # 8. Model Persistence (Integration with model_storage.py)
        # Saves the model checkpoint and tokenizer to the local filesystem
        from transformers import AutoModelForSeq2SeqLM
        from utils import get_tokenizer
        
        try:
            print(f"📦 Persisting model checkpoint for {dataset_name}...")
            # Load the tokenizer and model
            tokenizer = get_tokenizer("allenai/led-large-16384")
            model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384")
            
            # Define the specific output directory for the final trained checkpoint
            model_output_dir = "models"
            model_save_name = "final_mds_led"
            
            # Use our utility to save the checkpoint (config, weights, tokenizer)
            save_fine_tuned_transformer(model, tokenizer, output_dir=model_output_dir, model_name=model_save_name)
        except Exception as e:
            print(f"⚠️ Model persistence failed for {dataset_name}: {e}")

if __name__ == "__main__":
    main()
