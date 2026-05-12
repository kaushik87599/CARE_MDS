import os
import sys
import torch
import traceback
import time
import argparse
from datetime import datetime

def setup_environment():
    """
    Sets up the environment for the CARE_MDS pipeline.
    Ensures directories exist and dependencies are path-accessible.
    """
    print("🛠️ Setting up environment...")
    
    # Get absolute path to the root directory
    root_dir = os.path.abspath(os.path.dirname(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Ensure critical directories exist
    required_dirs = [
        "cache",
        "cache/cache",
        "cache/encoder_outputs",
        "cache/encoder_shards",
        "cache/fusion",
        "cache/entities",
        "cache/contradiction",
        "outputs",
        "outputs/generated_summaries",
        "models"
    ]
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
    
    # Download NLTK resources silently
    import nltk
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK resources ready.")
    except Exception as e:
        print(f"⚠️ Warning: NLTK download failed: {e}")
    
    print("✅ Environment setup complete.")

def run_phase(name, func, *args, **kwargs):
    """
    Helper to run a pipeline phase with timing and error handling.
    """
    print(f"\n{'#'*70}")
    print(f"🚀 STARTING {name}")
    print(f"{'#'*70}\n")
    
    start_time = time.time()
    try:
        func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"\n✅ {name} COMPLETED in {duration/60:.2f} minutes.")
        
        # Aggressive memory cleanup for Colab
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"\n❌ {name} FAILED.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc()
        # We raise the exception to stop the pipeline if a critical phase fails
        raise e

# Phase wrappers to keep main() clean
def phase1_wrapper():
    sys.path.insert(0, os.path.join(os.getcwd(), "preprocessing"))
    from preprocessing.analyze_dataset import main as preprocessing_main
    preprocessing_main()
    sys.path.pop(0)

def phase2_wrapper():
    sys.path.insert(0, os.path.join(os.getcwd(), "agent1"))
    from agent1.runner import run_agent1
    run_agent1()
    sys.path.pop(0)

def phase3_wrapper():
    from core_encoder.runner_encoder import run_core_encoder
    run_core_encoder()

def phase4_wrapper():
    from core_encoder.shard_outputs import run_sharding
    run_sharding()

def phase5_wrapper():
    from cross_document_fusion.fusion_engine import FusionEngine
    engine = FusionEngine()
    engine.run_full_pipeline()

def phase6_wrapper():
    from summary_generator.runner import Phase7Runner
    runner = Phase7Runner(do_eval=True)
    runner.run_all()

def main():
    parser = argparse.ArgumentParser(description="CARE_MDS Pipeline Runner")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], help="Run only a specific phase (1-6)")
    parser.add_argument("--start-at", type=int, choices=[1, 2, 3, 4, 5, 6], help="Start pipeline from a specific phase (1-6)")
    parser.add_argument("--list", action="store_true", help="List available phases")
    
    args = parser.parse_args()

    phases = [
        {"num": 1, "name": "PHASE 1: DATASET PREPROCESSING", "func": phase1_wrapper},
        {"num": 2, "name": "PHASE 2: AGENT 1 - CONTEXT PACKING", "func": phase2_wrapper},
        {"num": 3, "name": "PHASE 3: CORE ENCODER FORWARD PASS", "func": phase3_wrapper},
        {"num": 4, "name": "PHASE 4: SHARDING ENCODER OUTPUTS", "func": phase4_wrapper},
        {"num": 5, "name": "PHASE 5: CROSS-DOCUMENT FUSION ENGINE", "func": phase5_wrapper},
        {"num": 6, "name": "PHASE 6: SUMMARY GENERATION & EVALUATION", "func": phase6_wrapper},
    ]

    if args.list:
        print("\nAvailable Phases:")
        for p in phases:
            print(f"  {p['num']}: {p['name']}")
        return

    # Determine which phases to run
    run_list = []
    if args.phase:
        run_list = [p for p in phases if p["num"] == args.phase]
    elif args.start_at:
        run_list = [p for p in phases if p["num"] >= args.start_at]
    else:
        run_list = phases

    if not run_list:
        print("No valid phases selected to run.")
        return

    pipeline_start_time = time.time()
    print(f"{'='*70}")
    print(f"🌟 CARE_MDS: Multi-Document Summarization Pipeline 🌟")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Single Phase' if args.phase else 'Multi-Phase'}")
    print(f"{'='*70}")

    try:
        # Initial Setup (always run to ensure dirs exist)
        setup_environment()
        
        for p in run_list:
            run_phase(p["name"], p["func"])

        total_duration = (time.time() - pipeline_start_time) / 60
        print(f"\n{'='*70}")
        print("🎉 SELECTED PHASES COMPLETE!")
        print(f"Total time elapsed: {total_duration:.2f} minutes")
        print(f"{'='*70}")

    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.")
        sys.exit(1)
    except Exception:
        print("\n💥 Pipeline terminated due to errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
