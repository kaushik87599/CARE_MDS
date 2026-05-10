import os
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv

#check dataset
load_dotenv()

# os.environ["HF_TOKEN"] = ''

# The HF_TOKEN is now loaded into os.environ automatically

def check_dataset():
    datasets = {}
    
    # Path for Multi-News
    multi_news_path = 'datasets/multi_news_saved'
    if os.path.exists(multi_news_path):
        try:
            datasets['Multi-News'] = load_from_disk(multi_news_path)
            print(f"✅ Successfully loaded Multi-News dataset from {multi_news_path}")
        except Exception as e:
            print(f"❌ Error loading Multi-News: {e}")
    else:
        print(f"⚠️ Multi-News dataset not found at {multi_news_path}")

    # Path for CNN/Daily-News
    cnn_path = 'datasets/cnn_dailymail_saved'
    if os.path.exists(cnn_path):
        try:
            datasets['CNN/Daily-News'] = load_from_disk(cnn_path)
            print(f"✅ Successfully loaded CNN/Daily-News dataset from {cnn_path}")
        except Exception as e:
            print(f"❌ Error loading CNN/Daily-News: {e}")
    else:
        print(f"⚠️ CNN/Daily-News dataset not found at {cnn_path}")

    return datasets


