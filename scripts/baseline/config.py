import os
from datetime import datetime

#- Paths
#--Train
DATASET_PATH = "/Users/eli/research/uniner-finetune/datasets/dataset_8_9.csv"
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

#--Output
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_NAME = f"llama_baseline_{timestamp}"
NEW_MODEL_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(NEW_MODEL_DIR, exist_ok=True)
LOG_FILE = os.path.join(NEW_MODEL_DIR, "training_log.log")
