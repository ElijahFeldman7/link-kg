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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel

BASE_MODEL_NAME = "Universal-NER/UniNER-7B-all" 
ADAPTER_MODEL_PATH = "uniner_person_baseline"
DATASET_PATH = "dataset5.csv"
OUTPUT_DIR = "./uniner_continued_finetune"

TUPLE_DELIMITER = "{tuple_delimiter}"
RECORD_DELIMITER = "{record_delimiter}"
COMPLETION_DELIMITER = "{completion_delimiter}"


def create_interleaved_dataset(file_path: str) -> Dataset:

    try:
        df = pd.read_csv(file_path).fillna('')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    interleaved_data = []
    
    print("Preparing interleaved dataset...")
    for _, row in df.iterrows():
        input_text = row['Input_Text']
        if not isinstance(input_text, str) or not input_text.strip():
            continue

        person_labels = row['Person']
        if isinstance(person_labels, str) and person_labels.strip():
            person_list = [p.strip() for p in person_labels.split(',')]
            target_json = json.dumps(person_list)
            interleaved_data.append({
                "input_text": input_text,
                "target_text": target_json,
                "task_type": "person_only"
            })

        extracted_entities = row['Extracted_Entities']
        if isinstance(extracted_entities, str) and extracted_entities.strip():
            clean_target = extracted_entities.strip().replace(COMPLETION_DELIMITER, '').strip()
            interleaved_data.append({
                "input_text": input_text,
                "target_text": clean_target,
                "task_type": "full_extraction"
            })
            
    if not interleaved_data:
        print("Error: No valid data could be processed for training. Check CSV content.")
        exit()
        
    print(f"Created {len(interleaved_data)} total samples for interleaved training.")
    return Dataset.from_list(interleaved_data)


def setup_model_and_tokenizer():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    print(f"Loading existing adapters from: {ADAPTER_MODEL_PATH}")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_MODEL_PATH, is_trainable=True)
    except FileNotFoundError:
        print(f"Error: Adapter directory not found at '{ADAPTER_MODEL_PATH}'.")
        print("Please make sure the path is correct and contains files like 'adapter_model.bin' and 'adapter_config.json'.")
        exit()
    
    print("\nModel successfully loaded with previous adapters.")
    print("Trainable parameters after loading adapters:")
    model.print_trainable_parameters()
    
    return model, tokenizer


def preprocess_interleaved(example, tokenizer):
    task_type = example['task_type']
    input_text = example['input_text']
    target_output = example['target_text']

    if task_type == 'person_only':
        instruction = "From the text provided, extract all PERSON entities. Your output must be a JSON-formatted list of strings."
    else: 
        instruction = f"From the text provided, extract all entities. Format the output as a series of tuples. Each entity should be a tuple of ('entity', name, type, description). Separate each tuple with a '{RECORD_DELIMITER}'."

    full_prompt_text = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n{target_output}{tokenizer.eos_token}"
    
    result = tokenizer(
        full_prompt_text,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    
    result["labels"] = result["input_ids"].copy()
    
    prompt_only_for_masking = f"[INST] {instruction}\n\nText: {input_text} [/INST]\n"
    prompt_len = len(tokenizer(prompt_only_for_masking, add_special_tokens=False)['input_ids'])

    result["labels"][:prompt_len] = [-100] * prompt_len
        
    return result



def parse_structured_output(text: str) -> (list, bool):
    pattern = re.compile(
        r'\("entity"' + TUPLE_DELIMITER + r'(.*?)' + TUPLE_DELIMITER + r'(.*?)' + TUPLE_DELIMITER + r'.*?\)',
        re.DOTALL
    )
    try:
        matches = pattern.findall(text)
        entities = [(name.strip().upper(), type.strip().upper()) for name, type in matches]
        is_parsable = len(text.strip()) == 0 or len(entities) > 0
        return entities, is_parsable
    except Exception:
        return [], False

def compute_metrics_advanced(eval_pred, tokenizer):

    predictions_logits, labels = eval_pred
    predicted_ids = np.argmax(predictions_logits, axis=-1)
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    entity_stats = {} 
    error_stats = {'hallucination': 0, 'wrong_type': 0, 'missed_entity': 0}
    parsability_scores = []
    
    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        pred_entities, is_parsable = [], False
        label_entities = []

        if label_str.strip().startswith('['):
            try:
                label_list = json.loads(label_str)
                label_entities = [(name.strip().upper(), 'PERSON') for name in label_list]
            except json.JSONDecodeError: pass
            try:
                pred_list = json.loads(pred_str)
                if isinstance(pred_list, list):
                    pred_entities = [(str(name).strip().upper(), 'PERSON') for name in pred_list]
                is_parsable = True
            except json.JSONDecodeError: is_parsable = False
        else:
            pred_entities, is_parsable = parse_structured_output(pred_str)
            label_entities, _ = parse_structured_output(label_str)
        
        parsability_scores.append(1 if is_parsable else 0)
        
        pred_map = {name: type for name, type in pred_entities}
        label_map = {name: type for name, type in label_entities}
        
        for pred_name, pred_type in pred_map.items():
            if pred_name not in label_map:
                error_stats['hallucination'] += 1
            elif pred_type != label_map[pred_name]:
                error_stats['wrong_type'] += 1
        
        for label_name, label_type in label_map.items():
            if label_name not in pred_map:
                error_stats['missed_entity'] += 1
        
        pred_set = set(pred_entities)
        label_set = set(label_entities)
        all_types = {e_type for _, e_type in pred_set | label_set}

        for e_type in all_types:
            if e_type not in entity_stats:
                entity_stats[e_type] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            pred_set_type = {e for e in pred_set if e[1] == e_type}
            label_set_type = {e for e in label_set if e[1] == e_type}
            
            entity_stats[e_type]['tp'] += len(pred_set_type & label_set_type)
            entity_stats[e_type]['fp'] += len(pred_set_type - label_set_type)
            entity_stats[e_type]['fn'] += len(label_set_type - pred_set_type)

    final_metrics = {
        "parsability_score": np.mean(parsability_scores) if parsability_scores else 0.0,
        **error_stats
    }
    total_tp, total_fp, total_fn = 0, 0, 0

    for e_type, stats in entity_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        total_tp += tp; total_fp += fp; total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        final_metrics[f'precision_{e_type}'] = precision
        final_metrics[f'recall_{e_type}'] = recall
        final_metrics[f'f1_{e_type}'] = f1

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    final_metrics['f1_micro_avg'] = micro_f1

    return final_metrics


def main():
    """Main function to run the training and evaluation pipeline."""
    dataset = create_interleaved_dataset(DATASET_PATH)
    model, tokenizer = setup_model_and_tokenizer()
    
    tokenized_dataset = dataset.map(
        lambda x: preprocess_interleaved(x, tokenizer),
        batched=False,
        remove_columns=list(dataset.features)
    )
    
    splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_micro_avg",
        greater_is_better=True,
        gradient_checkpointing=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics_advanced(p, tokenizer),
    )

    print("\n--- Starting Continued Interleaved Fine-Tuning ---")
    trainer.train()
    print("\n--- Training Complete ---")

    print("\n--- Generating and Saving Test Predictions ---")
    test_results = trainer.predict(eval_dataset)
    
    predictions = np.argmax(test_results.predictions, axis=-1)
    labels = test_results.label_ids
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    output_prediction_file = os.path.join(OUTPUT_DIR, "test_predictions.txt")
    with open(output_prediction_file, "w") as writer:
        writer.write("--- Test Set Predictions and Labels ---\n\n")
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            writer.write(f"--- Example {i+1} ---\n")
            writer.write(f"GROUND TRUTH:\n{label.strip()}\n\n")
            writer.write(f"PREDICTED:\n{pred.strip()}\n")
            writer.write("="*20 + "\n\n")
    print(f"Predictions saved to {output_prediction_file}")

    print("\n--- Final Evaluation Results on Test Set ---")
    final_metrics = test_results.metrics
    print(json.dumps(final_metrics, indent=2))
    with open(os.path.join(OUTPUT_DIR, "final_eval_results.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    main()