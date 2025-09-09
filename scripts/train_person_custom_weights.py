import pandas as pd
import torch
import numpy as np
import re
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
MODEL_NAME = "Universal-NER/UniNER-7B-all"
DATASET_PATH = "dataset.csv" # Ensure this file is in the same directory
OUTPUT_DIR = "./uniner_person_granular_loss"

# --- Custom Delimiters ---
TUPLE_DELIMITER = "{tuple_delimiter}"
RECORD_DELIMITER = "{record_delimiter}"
COMPLETION_DELIMITER = "{completion_delimiter}"

# --- Granular Loss Weights (New Hierarchy) ---
ENTITY_NAME_WEIGHT = 10.0  # Highest weight for getting the name right
STRUCTURE_WEIGHT = 5.0   # Medium weight for the surrounding format
DESCRIPTION_WEIGHT = 1.0   # Lowest weight for the subjective description

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
try:
    df = pd.read_csv(DATASET_PATH, engine='python').dropna(subset=['Input_Text', 'Person', 'Extracted_Entities'])
    dataset = Dataset.from_pandas(df)
    print(f"Successfully loaded {len(dataset)} records from {DATASET_PATH}")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    exit()

# ==============================================================================
# 3. MODEL & TOKENIZER SETUP
# ==============================================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
print("\nModel setup complete:")
model.print_trainable_parameters()

# ==============================================================================
# 4. PREPROCESSING FUNCTION (WITH GRANULAR WEIGHT MAPPING)
# ==============================================================================
def preprocess_with_granular_weights(example):
    instruction = "From the text provided, extract all PERSON entities. Your output must be in the specified tuple format."
    input_text = example['Input_Text']
    person_names_str = example.get('Person', '')
    extracted_entities_str = example.get('Extracted_Entities', '')

    # Deconstruct the target string into components for precise weighting
    output_parts, tuple_components = [], []
    if isinstance(person_names_str, str) and person_names_str.strip():
        person_list = [p.strip() for p in person_names_str.split(',') if p.strip()]
        for person_name in person_list:
            description = "A person referenced in the text"
            pattern = re.compile(fr'(\"entity\"{re.escape(TUPLE_DELIMITER)}{re.escape(person_name)}{re.escape(TUPLE_DELIMITER)}PERSON{re.escape(TUPLE_DELIMITER)}(.*?)\\\)')
            match = pattern.search(extracted_entities_str)
            if match:
                description = match.group(1).strip()

            component = {
                "part1": f'(\"entity\"{TUPLE_DELIMITER}',
                "name": person_name,
                "part2": f'{TUPLE_DELIMITER}PERSON{TUPLE_DELIMITER}',
                "desc": description,
                "part3": ')'
            }
            tuple_components.append(component)
            output_parts.append(f'{component["part1"]}{component["name"]}{component["part2"]}{component["desc"]}{component["part3"]}')

    target_output = (RECORD_DELIMITER.join(output_parts) + RECORD_DELIMITER if output_parts else "") + COMPLETION_DELIMITER

    # Tokenize the full text
    full_prompt_text = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n{target_output}{tokenizer.eos_token}"
    result = tokenizer(full_prompt_text, truncation=True, max_length=1024, padding=False)
    result["labels"] = result["input_ids"].copy()
    
    # Create the weight map with a default value
    result["loss_weights"] = [1.0] * len(result["input_ids"])
    
    # Mask the prompt and set its weight to 0
    prompt_len = len(tokenizer(f"[INST] {instruction}\n\nText: {input_text} [/INST]\n", add_special_tokens=False)['input_ids'])
    result["labels"][:prompt_len] = [-100] * prompt_len
    result["loss_weights"][:prompt_len] = [0.0] * prompt_len
    
    # Apply the granular weights to the target tokens
    current_pos = prompt_len
    for i, component in enumerate(tuple_components):
        # Tokenize each part of the tuple separately to get token counts
        tokens = {key: tokenizer(value, add_special_tokens=False)['input_ids'] for key, value in component.items()}
        
        # Helper function to apply weights to a range of tokens
        def apply_weights(start, length, weight):
            for j in range(start, start + length):
                if j < len(result["loss_weights"]): result["loss_weights"][j] = weight
        
        # Apply weights based on our hierarchy
        apply_weights(current_pos, len(tokens["part1"])), STRUCTURE_WEIGHT)
        current_pos += len(tokens["part1"])
        
        apply_weights(current_pos, len(tokens["name"])), ENTITY_NAME_WEIGHT)
        current_pos += len(tokens["name"])
        
        apply_weights(current_pos, len(tokens["part2"])), STRUCTURE_WEIGHT)
        current_pos += len(tokens["part2"])
        
        apply_weights(current_pos, len(tokens["desc"])), DESCRIPTION_WEIGHT)
        current_pos += len(tokens["desc"])

        apply_weights(current_pos, len(tokens["part3"])), STRUCTURE_WEIGHT)
        current_pos += len(tokens["part3"])
        
        # Apply structure weight to the delimiter between records
        if i < len(tuple_components) - 1:
            tokens_rec_delim = tokenizer(RECORD_DELIMITER, add_special_tokens=False)['input_ids']
            apply_weights(current_pos, len(tokens_rec_delim), STRUCTURE_WEIGHT)
            current_pos += len(tokens_rec_delim)
            
    return result

# ==============================================================================
# 5. CUSTOM DATA COLLATOR
# ==============================================================================
class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors="pt"):
        # Extract loss_weights before they are removed by the parent collator
        loss_weights = [feature.pop("loss_weights") for feature in features]

        # Now, call the parent collator
        batch = super().__call__(features, return_tensors)

        # Pad the weights and add them to the batch
        max_length = batch["input_ids"].shape[1]
        padded_weights = []
        for weights in loss_weights:
            padded_weight = weights + [0.0] * (max_length - len(weights))
            padded_weights.append(padded_weight[:max_length])
        
        batch["loss_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        return batch

# ==============================================================================
# 6. CUSTOM WEIGHTED LOSS TRAINER
# ==============================================================================
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop("loss_weights")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        logits_flat = logits.view(-1, self.model.config.vocab_size)
        labels_flat = labels.view(-1)
        weights_flat = weights.view(-1)

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(logits_flat, labels_flat)

        weighted_loss = per_token_loss * weights_flat
        final_loss = weighted_loss.sum() / weights_flat.sum()
        
        return (final_loss, outputs) if return_outputs else final_loss

# ==============================================================================
# 7. EVALUATION METRICS
# ==============================================================================
def calculate_tuple_metrics(predictions, references):
    all_metrics = {"precision": [], "recall": [], "f1_score": []}
    tuple_pattern = re.compile(fr'(\"entity\"{re.escape(TUPLE_DELIMITER)}(.*?){re.escape(TUPLE_DELIMITER)}PERSON{re.escape(TUPLE_DELIMITER)}(.*?)\\\) ')

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
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    
    predicted_ids = np.argmax(logits, axis=-1)
    labels[labels == -100] = tokenizer.pad_token_id
    
    predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    label_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)

    processed_preds = [pred.split('[/INST]')[1].strip() if '[/INST]' in pred else "" for pred in predicted_strings]
    processed_labels = [label.split('[/INST]')[1].strip() if '[/INST]' in label else "" for label in label_strings]
    
    return calculate_tuple_metrics(processed_preds, processed_labels)

# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Process the dataset
    print("\n--- Preprocessing Dataset with Granular Weights ---")
    tokenized_dataset = dataset.map(preprocess_with_granular_weights, remove_columns=dataset.column_names)
    splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset, eval_dataset = splits["train"], splits["test"]
    print("--- Preprocessing Complete ---")

    # Instantiate the custom data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        num_train_epochs=25, # More complex loss may benefit from more epochs
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_score",
        greater_is_better=True,
    )

    # Instantiate our custom WeightedLossTrainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
    )

    # Start training
    print("\n--- Starting Training with Granular Weighted Loss Function ---")
    trainer.train()
    print("\n--- Training Complete ---")

    # Save the final model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n--- Best model saved to '{OUTPUT_DIR}' ---")
