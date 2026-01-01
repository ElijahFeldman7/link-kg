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
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

TUPLE_DELIMITER = "{tuple_delimiter}"
RECORD_DELIMITER = "{record_delimiter}"
COMPLETION_DELIMITER = "{completion_delimiter}"

try:
    df = pd.read_csv("dataset_8_9.csv").dropna(subset=['Input_Text', 'Person', 'Extracted_Entities'])
except (FileNotFoundError, KeyError) as e:
    print(f"Error: Could not read 'dataset_8_9.csv'. Make sure the file exists and contains the required columns. Details: {e}")
    exit()

dataset = Dataset.from_pandas(df[["Input_Text", "Person", "Extracted_Entities"]])

# --- Model and Tokenizer Setup (Unchanged) ---
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
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- REVISED: Preprocessing Function with Description Masking ---
def preprocess_structured_person(example):
    instruction = "From the text provided, extract all PERSON entities. Your output must be in the specified tuple format."
    input_text = example['Input_Text']
    
    person_names_str = example.get('Person', '')
    extracted_entities_str = example.get('Extracted_Entities', '')
    
    # --- Part 1: Construct the full target string (same as before) ---
    output_parts = []
    person_tuples_for_masking = [] # Store parts for masking

    if isinstance(person_names_str, str) and person_names_str.strip():
        person_list = [p.strip() for p in person_names_str.split(',')]
        
        for person_name in person_list:
            description = "A person referenced in the text"
            pattern = re.compile(
                f'\\("entity"\\{TUPLE_DELIMITER}{re.escape(person_name)}\\{TUPLE_DELIMITER}PERSON\\{TUPLE_DELIMITER}(.*?)\\)'
            )
            match = pattern.search(extracted_entities_str)
            if match:
                description = match.group(1).strip()
            
            # Store the structured parts for later use in masking
            person_tuples_for_masking.append({
                "part1": f'("entity"{TUPLE_DELIMITER}{person_name}{TUPLE_DELIMITER}PERSON{TUPLE_DELIMITER}',
                "desc": description,
                "part3": ')'
            })
            
            output_parts.append(f'{person_tuples_for_masking[-1]["part1"]}{person_tuples_for_masking[-1]["desc"]}{person_tuples_for_masking[-1]["part3"]}')

    if output_parts:
        target_output = RECORD_DELIMITER.join(output_parts) + RECORD_DELIMITER + COMPLETION_DELIMITER
    else:
        target_output = COMPLETION_DELIMITER

    # --- Part 2: Tokenize and create initial labels (same as before) ---
    full_prompt_text = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n{target_output}{tokenizer.eos_token}"
    result = tokenizer(full_prompt_text, truncation=True, max_length=512, padding="max_length")
    result["labels"] = result["input_ids"].copy()

    prompt_only_for_masking = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n"
    prompt_len = len(tokenizer(prompt_only_for_masking, add_special_tokens=False)['input_ids'])
    result["labels"][:prompt_len] = [-100] * prompt_len
    
    current_pos = prompt_len
    for i, p_tuple in enumerate(person_tuples_for_masking):
        tokens_part1 = tokenizer(p_tuple["part1"], add_special_tokens=False)['input_ids']
        tokens_desc = tokenizer(p_tuple["desc"], add_special_tokens=False)['input_ids']
        tokens_part3 = tokenizer(p_tuple["part3"], add_special_tokens=False)['input_ids']
        tokens_rec_delim = tokenizer(RECORD_DELIMITER, add_special_tokens=False)['input_ids']

        current_pos += len(tokens_part1)
        
        desc_start = current_pos
        desc_end = current_pos + len(tokens_desc)
        
        if desc_end <= len(result["labels"]):
             result["labels"][desc_start:desc_end] = [-100] * len(tokens_desc)
        
        current_pos += len(tokens_desc) + len(tokens_part3)
        
        if i < len(person_tuples_for_masking) - 1:
            current_pos += len(tokens_rec_delim)
            
    return result
tokenized_dataset = dataset.map(preprocess_structured_person, remove_columns=list(dataset.features))
splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset, eval_dataset = splits["train"], splits["test"]

def calculate_tuple_metrics(predictions, references):
    all_metrics = {"precision": [], "recall": [], "f1_score": []}
    
    tuple_pattern = re.compile(f'\\("entity"\\{TUPLE_DELIMITER}(.*?)\\{TUPLE_DELIMITER}PERSON\\{TUPLE_DELIMITER}(.*?)\\)')

    for pred_str, ref_str in zip(predictions, references):
        pred_tuples = tuple_pattern.findall(pred_str)
        ref_tuples = tuple_pattern.findall(ref_str)
        
        pred_set = set((name.strip().upper(), desc.strip().upper()) for name, desc in pred_tuples)
        ref_set = set((name.strip().upper(), desc.strip().upper()) for name, desc in ref_tuples)
        
        true_positives = len(pred_set.intersection(ref_set))
        
        precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = true_positives / len(ref_set) if len(ref_set) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        all_metrics["precision"].append(precision)
        all_metrics["recall"].append(recall)
        all_metrics["f1_score"].append(f1)

    return {key: np.mean(values) for key, values in all_metrics.items()}

def compute_metrics_wrapper(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_ids = np.argmax(predictions, axis=-1)
    labels[labels == -100] = tokenizer.pad_token_id
    
    predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    label_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up the generated text to only include the response
    processed_preds = [pred.split('[/INST]')[1].strip() if '[/INST]' in pred else "" for pred in predicted_strings]
    processed_labels = [label.split('[/INST]')[1].strip() if '[/INST]' in label else "" for label in label_strings]
    
    return calculate_tuple_metrics(processed_preds, processed_labels)

# --- Training Arguments (Unchanged, but output_dir updated) ---
training_args = TrainingArguments(
    output_dir="./uniner_person_structured",
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

# --- Trainer Setup (Updated with new compute_metrics) ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_wrapper,
)

# --- Run Training ---
print("--- Starting Structured PERSON Extraction Training ---")
trainer.train()
print("\n--- Training Complete ---")