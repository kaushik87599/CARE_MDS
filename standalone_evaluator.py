import os
import json
import sys
import pandas as pd
from tqdm import tqdm

# IMPORTANT: If running from outside the main CARE_MDS directory, uncomment and set this path
# sys.path.insert(0, "/path/to/CARE_MDS")
from summary_generator.evaluator import SummaryEvaluator

def run_standalone_evaluation(results_dir: str):
    """
    Evaluates existing summary JSONs without re-generating text.
    
    """
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return

    # 1. Figure out exactly which files we have
    files = [f for f in os.listdir(results_dir) if f.endswith("_results.json")]
    print(f"📂 Found {len(files)} result files in {results_dir}.")
    
    if len(files) == 0:
        return
        
    cluster_ids_to_process = set([f.replace("_results.json", "") for f in files])
    
    # 2. Build ground truth ONLY for these specific clusters
    print("⏳ Loading Multi-News dataset to build ground truth (this might take a minute)...")
    from agent1.utils.load_multi_dataset import load_multi_dataset
    from agent1.sentence_splitting import document_split, sentence_split
    
    df = load_multi_dataset()
    ground_truth_data = {}
    
    print("⏳ Extracting references and source sentences (Skipping missing outputs)...")
    # Wrap in tqdm to show extraction progress
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Ground Truth"):
        cluster_id = str(idx)
        
        # SKIP if we don't have an output for it!
        if cluster_id not in cluster_ids_to_process:
            continue
            
        reference = row.get("summary", "")
        if pd.isna(reference):
            reference = ""
            
        document_row = row.get("document", "")
        source_sentences = []
        if pd.notna(document_row) and str(document_row).strip() != "":
            documents = document_split(document_row)
            for doc in documents:
                sents = sentence_split(doc)
                source_sentences.extend(sents)
                
        ground_truth_data[cluster_id] = {
            "reference": reference,
            "source_sentences": source_sentences
        }
        
    print(f"✅ Ground truth dictionary built for {len(ground_truth_data)} existing clusters.")

    # 3. Initialize models
    print("\n🚀 Initializing Evaluator Models...")
    evaluator = SummaryEvaluator()
    
    print("⏳ Pre-warming models (triggering initial downloads, please wait)...")
    # We do a dummy run so that BERTScore downloads its weights BEFORE the progress bar starts
    _ = evaluator.evaluate(
        summary="This is a test.",
        reference="This is a test.",
        source_sentences=["This is a test."],
        contradictions=[],
        entities=[]
    )
    print("✅ Models fully loaded and ready!\n")
    
    # 4. Run the evaluations
    updated_count = 0
    for filename in tqdm(files, desc="Evaluating Summaries"):
        cluster_id = filename.replace("_results.json", "")
        file_path = os.path.join(results_dir, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if data.get("evaluation") is not None:
            continue
            
        ground_truth = ground_truth_data.get(cluster_id)
        if not ground_truth:
            continue
            
        try:
            eval_results = evaluator.evaluate(
                summary=data.get("generated_summary", ""),
                reference=ground_truth["reference"],
                source_sentences=ground_truth["source_sentences"],
                contradictions=data.get("contradiction_metadata", {}).get("signals", []),
                entities=data.get("entity_metadata", {}).get("entities", [])
            )
            
            data["evaluation"] = eval_results
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                
            updated_count += 1
        except Exception as e:
            print(f"\n❌ Error evaluating {cluster_id}: {e}")
            
    print(f"\n🎉 Evaluation complete! Successfully updated {updated_count} files.")


if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION FOR COLAB
    # ==========================================
    
    # 1. Provide the path where your JSON outputs are stored
    MY_RESULTS_DIR = "/content/drive/MyDrive/CARE_MDS/outputs/generated_summaries" 
    
    # 2. Run the pipeline
    run_standalone_evaluation(MY_RESULTS_DIR)
