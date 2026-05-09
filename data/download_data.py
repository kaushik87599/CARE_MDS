import os
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# The HF_TOKEN is now loaded into os.environ automatically

cnn = load_dataset("abisee/cnn_dailymail", "3.0.0")
multi_news = load_dataset("alexfabbri/multi_news", "3.0.0")

cnn.save_to_disk("./datasets/cnn_dailymail_saved")
multi_news.save_to_disk("./datasets/multi_news_saved")

# print(cnn["train"][0])
# print(multi_news["train"][0])
