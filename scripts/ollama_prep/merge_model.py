import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

base_model_name = "Universal-NER/UniNER-7B-all"
adapter_path = "./uniner_person_baseline"
merged_model_path = "./merged_uniner_model"

print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    return_dict=True,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, adapter_path)

model = model.merge_and_unload()

print(f"\nSaving merged model to {merged_model_path}...")
os.makedirs(merged_model_path, exist_ok=True)
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("\n done!")