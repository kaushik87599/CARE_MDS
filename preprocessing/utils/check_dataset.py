import os
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_dataset(ROOT_PATH=None):
    if ROOT_PATH is None:
        ROOT_PATH = os.getenv("DATASETS_DIR", "datasets")
        
    datasets = {}
    
    # Path for Multi-News
    multi_news_path = os.path.join(ROOT_PATH, 'multi_news_saved')
    if os.path.exists(multi_news_path):
        try:
            datasets['Multi-News'] = load_from_disk(multi_news_path)
            print(f"✅ Loaded Multi-News from {multi_news_path}")
        except Exception as e:
            print(f"❌ Error loading Multi-News: {e}")
    else:
        print(f"⚠️ Multi-News dataset not found at {multi_news_path}")
    
    # Path for CNN/Daily-News
    cnn_path = os.path.join(ROOT_PATH, 'cnn_dailymail_saved')
    if os.path.exists(cnn_path):
        try:
            datasets['CNN/Daily-News'] = load_from_disk(cnn_path)
            print(f"✅ Loaded CNN/Daily-News from {cnn_path}")
        except Exception as e:
            print(f"❌ Error loading CNN/Daily-News: {e}")
    else:
        print(f"⚠️ CNN/Daily-News dataset not found at {cnn_path}")

    return datasets
