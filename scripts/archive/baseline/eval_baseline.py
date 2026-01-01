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
)
from tqdm import tqdm

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "dataset5.csv"
OUTPUT_DIR = "./uniner_baseline_evaluation_with_scores"

TUPLE_DELIMITER = "{tuple_delimiter}"
RECORD_DELIMITER = "{record_delimiter}"
COMPLETION_DELIMITER = "{completion_delimiter}"

def create_dataset_from_output_col(file_path: str) -> Dataset:
    try:
        df = pd.read_csv(file_path).fillna('')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    processed_data = []
    print("Preparing dataset from 'Output' column...")
    for _, row in df.iterrows():
        input_text = row.get('Input_Text')
        target_output = row.get('Output')

        if not (isinstance(input_text, str) and input_text.strip()):
            continue
        if not (isinstance(target_output, str) and target_output.strip()):
            continue
        
        clean_target = target_output.strip()

        processed_data.append({
            "input_text": input_text,
            "ground_truth": clean_target,
        })

    if not processed_data:
        print("Error: No valid data could be processed. Check CSV content.")
        exit()

    print(f"Created {len(processed_data)} samples for evaluation.")
    return Dataset.from_list(processed_data)

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
        tokenizer.padding_side = "left"

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model.eval()
    return model, tokenizer

def create_inference_prompt(input_text: str) -> str:
    instruction = f"""From the text below, extract all relevant entities and their relationships based on these rules:

1.  **Entity Extraction**:
    * **Format**: ("entity"{TUPLE_DELIMITER}NAME{TUPLE_DELIMITER}TYPE{TUPLE_DELIMITER}description)

2.  **Relationship Extraction**:
    * **Format**: ("relationship"{TUPLE_DELIMITER}ENTITY_1{TUPLE_DELIMITER}ENTITY_2{TUPLE_DELIMITER}description{TUPLE_DELIMITER}strength_score_0_to_10)

**Output Rules**:
* Extract only information explicitly stated in the text.
* Separate every tuple with {RECORD_DELIMITER}.
* End the final output with {COMPLETION_DELIMITER}.

**Text to Analyze**: {input_text}"""

    return f"[INST] {instruction} [/INST]\n"

def parse_for_eval(text: str) -> (set, dict, bool):
    entities = set()
    relationships = {} 

    text = text.replace("</s>", "").strip()

    entity_pattern = re.compile(
        r'\("entity"' + re.escape(TUPLE_DELIMITER) + r'(.*?)' + re.escape(TUPLE_DELIMITER) + r'(.*?)' + re.escape(TUPLE_DELIMITER) + r'.*?\)', re.DOTALL
    )
    rel_pattern_with_score = re.compile(
        r'\("relationship"' + re.escape(TUPLE_DELIMITER) + r'(.*?)' + re.escape(TUPLE_DELIMITER) + r'(.*?)' + re.escape(TUPLE_DELIMITER) + r'.*?' + re.escape(TUPLE_DELIMITER) + r'(\d+)\s*\)', re.DOTALL
    )

    try:
        for name, type in entity_pattern.findall(text):
            entities.add((name.strip().upper(), type.strip().upper()))
        
        for ent1, ent2, score in rel_pattern_with_score.findall(text):
            sorted_ents = tuple(sorted([ent1.strip().upper(), ent2.strip().upper()]))
            relationships[sorted_ents] = int(score)

        is_parsable = len(text.strip()) == 0 or (len(entities) > 0 or len(relationships) > 0)
        return entities, relationships, is_parsable
    except Exception:
        return set(), {}, False

def compute_metrics(predictions: list, ground_truths: list):
    parsability_scores, score_errors = [], []
    entity_tp, entity_fp, entity_fn = 0, 0, 0
    rel_tp, rel_fp, rel_fn = 0, 0, 0

    for pred_str, label_str in zip(predictions, ground_truths):
        pred_entities, pred_rels, is_parsable = parse_for_eval(pred_str)
        label_entities, label_rels, _ = parse_for_eval(label_str)

        parsability_scores.append(1 if is_parsable else 0)

        entity_tp += len(pred_entities & label_entities)
        entity_fp += len(pred_entities - label_entities)
        entity_fn += len(label_entities - pred_entities)

        pred_rel_keys = set(pred_rels.keys())
        label_rel_keys = set(label_rels.keys())
        rel_tp += len(pred_rel_keys & label_rel_keys)
        rel_fp += len(pred_rel_keys - label_rel_keys)
        
        rel_fn += len(label_rel_keys - pred_rel_keys)
        
        true_positive_rels = pred_rel_keys & label_rel_keys
        for rel_pair in true_positive_rels:
            if rel_pair in pred_rels and rel_pair in label_rels:
                error = abs(pred_rels[rel_pair] - label_rels[rel_pair])
                score_errors.append(error)

    final_metrics = {}
    final_metrics["parsability_score"] = np.mean(parsability_scores) if parsability_scores else 0.0

    entity_precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0.0
    entity_recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0.0
    final_metrics['entity_f1'] = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
    
    rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0.0
    rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0.0
    final_metrics['relationship_f1'] = 2 * (rel_precision * rel_recall) / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0
    
    final_metrics['relationship_score_mae'] = np.mean(score_errors) if score_errors else 0.0

    return final_metrics

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model, tokenizer = setup_model_and_tokenizer()
    eval_dataset = create_dataset_from_output_col(DATASET_PATH)

    predictions = []
    ground_truths = []
    
    eos_token_str = tokenizer.eos_token
    
    print("\nRunning inference on the dataset...")
    with torch.no_grad():
        for example in tqdm(eval_dataset):
            prompt = create_inference_prompt(example["input_text"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            decoded_output = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
            
            if eos_token_str:
                decoded_output = decoded_output.replace(eos_token_str, "").strip()
            
            predictions.append(decoded_output.strip())
            ground_truths.append(example["ground_truth"])

    output_prediction_file = os.path.join(OUTPUT_DIR, "baseline_predictions.txt")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        for i, (pred, label, inp) in enumerate(zip(predictions, ground_truths, eval_dataset['input_text'])):
            writer.write(f"--- Example {i+1} ---\n")
            writer.write(f"INPUT TEXT:\n{inp.strip()}\n\n")
            writer.write(f"GROUND TRUTH:\n{label.strip()}\n\n")
            writer.write(f"PREDICTED:\n{pred.strip()}\n")
            writer.write("="*20 + "\n\n")
    print(f"\nRaw predictions saved to {output_prediction_file}")

    final_metrics = compute_metrics(predictions, ground_truths)
    
    print("\n--- BASELINE EVALUATION RESULTS ---")
    print(json.dumps(final_metrics, indent=2))

    output_metrics_file = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
    with open(output_metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to {output_metrics_file}")

if __name__ == "__main__":
    main()