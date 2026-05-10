import os
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# The HF_TOKEN is now loaded into os.environ automatically

cnn = load_dataset("abisee/cnn_dailymail", "3.0.0")
# Downsample CNN/DailyMail: Train 8k, Val 1k, Test 1k
cnn['train'] = cnn['train'].select(range(min(8000, len(cnn['train']))))
cnn['validation'] = cnn['validation'].select(range(min(1000, len(cnn['validation']))))
cnn['test'] = cnn['test'].select(range(min(1000, len(cnn['test']))))

multi_news = load_dataset("alexfabbri/multi_news", "3.0.0")
# Downsample Multi-News: Train 4k, Val 500, Test 500
multi_news['train'] = multi_news['train'].select(range(min(4000, len(multi_news['train']))))
multi_news['validation'] = multi_news['validation'].select(range(min(500, len(multi_news['validation']))))
multi_news['test'] = multi_news['test'].select(range(min(500, len(multi_news['test']))))

cnn.save_to_disk("./datasets/cnn_dailymail_saved")
multi_news.save_to_disk("./datasets/multi_news_saved")

# print(cnn["train"][0])
# print(multi_news["train"][0])
