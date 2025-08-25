import pandas as pd
import torch
import numpy as np
import json
import re
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

try:
    df = pd.read_csv("dataset1.csv").dropna(subset=['Input_Text', 'Person'])
    df = df.rename(columns={"Person": "labels"})
except (FileNotFoundError, KeyError) as e:
    print(f"Error: Could not read 'dataset.csv'. Make sure the file exists and contains 'text' and 'Person' columns. Details: {e}")
    exit()

dataset = Dataset.from_pandas(df[["Input_Text", "labels"]])

model_name = "Universal-NER/UniNER-7B-all"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def preprocess_simple(example):
    instruction = "From the text provided, extract all PERSON entities. Your output must be a JSON-formatted list of strings."
    input_text = example['Input_Text']
    
    try:
        if isinstance(example['labels'], str) and example['labels'].strip():
            person_list = [p.strip() for p in example['labels'].split(',')]
            target_output = json.dumps(person_list)
        else:
            target_output = "[]"
    except:
        target_output = "[]"

    full_prompt_text = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n{target_output}{tokenizer.eos_token}"

    result = tokenizer(
        full_prompt_text,
        truncation=True,
        max_length=512,  
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()

    prompt_only_for_masking = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n"
    prompt_only_tokens = tokenizer(prompt_only_for_masking, add_special_tokens=True)
    prompt_len = len(prompt_only_tokens['input_ids'])

    for i in range(prompt_len):
        if i < len(result["labels"]):
            result["labels"][i] = -100
        
    return result

tokenized_dataset = dataset.map(preprocess_simple, remove_columns=list(dataset.features))

splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]
test_dataset = splits["test"]

def calculate_list_metrics(predictions, references):
    all_metrics = {"precision": [], "recall": [], "f1_score": []}
    
    for pred_str, ref_str in zip(predictions, references):
        try:
            pred_list = json.loads(pred_str.strip())
            if not isinstance(pred_list, list) or not all(isinstance(i, str) for i in pred_list):
                 pred_list = [] 
        except (json.JSONDecodeError, TypeError):
            pred_list = [] 

        try:
            ref_list = json.loads(ref_str.strip())
        except (json.JSONDecodeError, TypeError):
            ref_list = []

        pred_set = set(p.upper() for p in pred_list)
        ref_set = set(r.upper() for r in ref_list)

        true_positives = len(pred_set.intersection(ref_set))
        
        precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = true_positives / len(ref_set) if len(ref_set) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        all_metrics["precision"].append(precision)
        all_metrics["recall"].append(recall)
        all_metrics["f1_score"].append(f1)

    avg_metrics = {key: np.mean(values) if values else 0.0 for key, values in all_metrics.items()}
    return avg_metrics


def compute_metrics_wrapper(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_ids = np.argmax(predictions, axis=-1)
    
    labels_copy = labels.copy()
    predicted_ids_copy = predicted_ids.copy()

    labels_copy[labels_copy == -100] = tokenizer.pad_token_id

    processed_predicted_strings = []
    processed_label_strings = []

    for i in range(labels_copy.shape[0]):
        start_index = 0
        for j in range(labels_copy.shape[1]):
            if labels[i, j] != -100: 
                start_index = j
                break
        
        sliced_predicted_ids = predicted_ids_copy[i, start_index:]
        sliced_label_ids = labels_copy[i, start_index:]

        processed_predicted_strings.append(tokenizer.decode(sliced_predicted_ids, skip_special_tokens=True))
        processed_label_strings.append(tokenizer.decode(sliced_label_ids, skip_special_tokens=True))
    
    return calculate_list_metrics(processed_predicted_strings, processed_label_strings)

training_args = TrainingArguments(
    output_dir="./uniner_person_baseline",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_score",
    greater_is_better=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_wrapper,
)

print("--- Starting Simplified Baseline Training (PERSON Extraction Only) ---")
trainer.train()
print("\n--- Training Complete ---")

print("\n--- Evaluating on the Final Test Set and Saving Predictions... ---")

original_splits = dataset.train_test_split(test_size=0.2, seed=42)
raw_test_dataset = original_splits["test"]

instruction = "From the text provided, extract all PERSON entities. Your output must be a JSON-formatted list of strings."
prompts = []
for example in raw_test_dataset:
    input_text = example['Input_Text']
    prompt = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n"
    prompts.append(prompt)



predicted_strings = []
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    predicted_strings.append(answer)

label_strings = []
for example in raw_test_dataset:
    try:
        if isinstance(example['labels'], str) and example['labels'].strip():
            person_list = [p.strip() for p in example['labels'].split(',')]
            target_output = json.dumps(person_list)
        else:
            target_output = "[]"
    except:
        target_output = "[]"
    label_strings.append(target_output)

output_dir = "./uniner_person_baseline"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/test_predictions_simple.txt", "w") as f:
    for i, (pred, label) in enumerate(zip(predicted_strings, label_strings)):
        f.write(f"--- Example {i+1} ---\n")
        f.write(f"GROUND TRUTH:\n{label.strip()}\n")
        f.write(f"PREDICTED:\n{pred.strip()}\n")
        f.write("="*20 + "\n\n")

final_metrics = calculate_list_metrics(predicted_strings, label_strings)
with open(f"{output_dir}/test_results_simple.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n--- Baseline model and predictions saved to '{output_dir}' ---")
print("\n--- Final Test Results: ---")
print(json.dumps(final_metrics, indent=2))
