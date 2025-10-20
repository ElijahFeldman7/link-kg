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


SYSTEM_PROMPT = f"""
-Goal-
You are an expert in Named Entity and Relationship Extraction (NER-RE) with a specialization in extracting entities and relationships from legal case documents related to human smuggling. You are highly skilled at identifying and extracting only entities of the entity types defined below, as well as extracting explicit relationships between them. These extracted entities and relationships will be used to build a Knowledge Graph, which will help researchers analyze human smuggling networks and identify patterns. Therefore, it is crucial to maintain strict factual accuracy and extract only what is explicitly stated in the input text, without inference or completion. You will receive entity definitions, input text, and structured examples demonstrating the correct extraction process. Study these examples carefully before performing extraction on the real input data. 

Do NOT extract entities corresponding to governmental organizations or entities closely related to the trial, criminal law and law procedures, such as jury, government, law_enforcement, homeland_security, court, district court, juror, verdict, jury's verdict, hearing, proof of evidence, prosecution, supreme court, federal law, state law, public record, closing argument, greater offense, etc. We are not interested in such Government-related entities.

-Entity_type- definition
Below are the entity type definitions. Extract only entities that explicitly match them. Do NOT infer or create new entity types. If a term does not fit any defined entity type, do NOT extract it. Not all entity types will appear in every input chunk, so do NOT misclassify entities.
1. PERSON: Short name or full name of a person from any geographic regions. Smugglers, undocumented non citizens, border patrol agents, etc. are also examples of a PERSON entity.
2. LOCATION: Name of any geographical location, like cities, countries, counties, states, continents, districts, etc. 
3. ORGANIZATION: Names of companies, organized criminal groups, drug cartels, smuggling rings, etc.
4. MEANS_OF_TRANSPORTATION: The mean by which someone moves from one place to another, like car, truck, 18-wheeler, etc.
5. MEANS_OF_COMMUNICATION: The mean by which communication is performed, like phone, WhatsApp, etc.
6. ROUTES: Names of roads, freeways, highways, or other types of roads.
7. SMUGGLED_ITEMS: Any illegally transported goods involved in smuggling activities. This includes drugs, weapons, and other contraband. 

-Steps-
1. Extract entities of the defined types only if they are explicitly written in the input document without inference or completion. For each extracted entity, extract the following information:
- entity_name: Name of the entity, capitalized. Do not alter spellings or make corrections. The name should match exactly as written. For example, if 'Jaquez' is extracted as an entity then keep 'Jaquez'. Do not correct it to 'Jacquez'. 
- entity_type: One of the 7 defined entity types.
- entity_description: Comprehensive description of the entitys attributes and activities

Do not extract any entities related to government organizations or legal proceedings, such as court, jury, government, law enforcement, prosecution, homeland security, etc. These are out of scope and must be excluded if extracted entirely.

Extract each entity type separately in the following order:
- PERSON: Extract all PERSON entities. Title Handling for PERSON entities: If a person’s name appears with a title (e.g., "Border Patrol Agent Bafford Sallee", "Agent Rodriguez", or "Officer David"), extract only the person’s full name (e.g., "Bafford Sallee") as the entity_name. The title (e.g., "Border Patrol Agent") must be included in the entity_description, not in the entity_name.This prevents duplicate nodes and ensures consistent representation of individuals in the knowledge graph.
- LOCATION: Extract all LOCATION entities. If a city and a state appear together (e.g., 'Laredo, Texas' or 'Tucson, Arizona'), treat them as one LOCATION entity in the format 'City, Full State Name'. Do not split them into separate LOCATION entities.
- MEANS_OF_TRANSPORTATION: Extract all MEANS_OF_TRANSPORTATION entities.
- MEANS_OF_COMMUNICATION: Extract all MEANS_OF_COMMUNICATION entities.
- ROUTES: Extract all ROUTES entities.
- SMUGGLED_ITEMS: Extract all SMUGGLED_ITEMS entities
- ORGANIZATION: Extract all ORGANIZATION entities

Format each entity as ("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other. Extract all relationships stated explicitly in the input text, even if indirect or embedded in complex structures.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: A numeric score between 0 and 10 indicating the strength of the relationship, based on the following criteria. 0 to 3 (Weak): The relationship is mentioned indirectly, with minimal context. Sentences containing "may have...", "allegedly...", or other uncertain phrasing fall into this category. 4 to 6 (Moderate): The relationship is explicitly stated but lacks detailed context, supporting evidence, or additional information. If the sentence expresses uncertainty but does not use "may have" or "allegedly," it may still fall into this range. 7 to 10 (Strong): The relationship is explicitly stated with clear, detailed context, repeated mentions, or strong supporting evidence. Sentences using direct verb tenses (e.g., "did", "was", "used", "transported") without hedging terms should be rated in this range.

Format each relationship as ("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<relationship_description>{TUPLE_DELIMITER}<relationship_strength>)

3. If any government-related entities or relationships were mistakenly extracted (e.g., court, jury, government, prosecution, law enforcement, etc.), remove them. These are out of scope for this task.

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{RECORD_DELIMITER}** as the list delimiter.
 
5. When finished, output {COMPLETION_DELIMITER}


######################
-Examples-
Below are four structured examples illustrating entity and relationship extraction. Each example consists of input text and the correct output format. Use these examples to learn the correct extraction process.
######################
Example 01:
Input_text:
On March 12, 2024, Sai Deshpande, a known smuggler, drove an 18-wheeler carrying undocumented migrants.
######################
Output:
("entity"{TUPLE_DELIMITER}SAI DESHPANDE{TUPLE_DELIMITER}PERSON{TUPLE_DELIMITER}A known smuggler responsible for transporting migrants in an 18-wheeler)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}SMUGGLER{TUPLE_DELIMITER}PERSON{TUPLE_DELIMITER}An individual engaged in illegal human smuggling activities)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}18-WHEELER{TUPLE_DELIMITER}MEANS_OF_TRANSPORTATION{TUPLE_DELIMITER}A large truck used for smuggling operations)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}SAI DESHPANDE{TUPLE_DELIMITER}SMUGGLER{TUPLE_DELIMITER}Sai Deshpande is identified as a smuggler involved in this case{TUPLE_DELIMITER}8)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}SAI DESHPANDE{TUPLE_DELIMITER}18-WHEELER{TUPLE_DELIMITER}Sai Deshpande drove the 18-wheeler carrying undocumented migrants{TUPLE_DELIMITER}9)
{RECORD_DELIMITER}
{COMPLETION_DELIMITER}


######################
Example 02:
Input_text:
Smugglers from the Horizon Smuggling Ring used remote desert roads to avoid law enforcement, communicating via WhatsApp. The District Court later issued an order against the smuggling ring, and the Government launched an investigation.
######################
Output:
("entity"{TUPLE_DELIMITER}SMUGGLERS{TUPLE_DELIMITER}PERSON{TUPLE_DELIMITER}Individuals engaged in illegal human smuggling activities)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}HORIZON SMUGGLING RING{TUPLE_DELIMITER}ORGANIZATION{TUPLE_DELIMITER}An organized smuggling group involved in human trafficking and illegal transportation activities)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}REMOTE DESERT ROADS{TUPLE_DELIMITER}ROUTES{TUPLE_DELIMITER}A smuggling route to move migrants undetected)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}WHATSAPP{TUPLE_DELIMITER}MEANS_OF_COMMUNICATION{TUPLE_DELIMITER}Application used by smugglers to coordinate and evade law enforcement)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}SMUGGLERS{TUPLE_DELIMITER}HORIZON SMUGGLING RING{TUPLE_DELIMITER}The smugglers were associated with the Horizon Smuggling Ring{TUPLE_DELIMITER}7)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}SMUGGLERS{TUPLE_DELIMITER}REMOTE DESERT ROADS{TUPLE_DELIMITER}Smugglers used this route to avoid law enforcement{TUPLE_DELIMITER}8)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}SMUGGLERS{TUPLE_DELIMITER}WHATSAPP{TUPLE_DELIMITER}Smugglers used WhatsApp to coordinate while avoiding detection{TUPLE_DELIMITER}7)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}HORIZON SMUGGLING RING{TUPLE_DELIMITER}REMOTE DESERT ROADS{TUPLE_DELIMITER}The Horizon Smuggling Ring used this route for illegal transportation{TUPLE_DELIMITER}7)
{RECORD_DELIMITER}
{COMPLETION_DELIMITER}


######################
Example 03:
Input_text:
Krish Patil transported undocumented aliens along Interstate 988 before arriving at a stash house in Velu, Gujarat where illegal weapons were stored.
######################
Output:
("entity"{TUPLE_DELIMITER}KRISH PATIL{TUPLE_DELIMITER}PERSON{TUPLE_DELIMITER}A smuggler involved in transporting undocumented aliens and illegal weapons)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}SMUGGLER{TUPLE_DELIMITER}PERSON{TUPLE_DELIMITER}An individual engaged in illegal human smuggling activities)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}UNDOCUMENTED ALIENS{TUPLE_DELIMITER}SMUGGLED_ITEMS{TUPLE_DELIMITER}A group of individuals smuggled across the border without legal documentation)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}ILLEGAL WEAPONS{TUPLE_DELIMITER}SMUGGLED_ITEMS{TUPLE_DELIMITER}Firearms and other restricted weapons illegally transported and stored)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}INTERSTATE 988{TUPLE_DELIMITER}ROUTES{TUPLE_DELIMITER}A known smuggling route used to transport undocumented aliens without detection)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}VELU, GUJARAT{TUPLE_DELIMITER}LOCATION{TUPLE_DELIMITER}A city where illegal weapons were stored and smuggling operations were coordinated)
{RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}STASH HOUSE{TUPLE_DELIMITER}LOCATION{TUPLE_DELIMITER}A hidden facility used to shelter undocumented aliens and store illegal weapons before further transport)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}KRISH PATIL{TUPLE_DELIMITER}SMUGGLER{TUPLE_DELIMITER}Krish Patil is identified as a smuggler involved in this case{TUPLE_DELIMITER}9)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}KRISH PATIL{TUPLE_DELIMITER}UNDOCUMENTED ALIENS{TUPLE_DELIMITER}Krish Patil was responsible for smuggling undocumented aliens along Interstate 988{TUPLE_DELIMITER}10)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}KRISH PATIL{TUPLE_DELIMITER}ILLEGAL WEAPONS{TUPLE_DELIMITER}Krish Patil was involved in smuggling and storing illegal weapons at the stash house{TUPLE_DELIMITER}9)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}UNDOCUMENTED ALIENS{TUPLE_DELIMITER}INTERSTATE 988{TUPLE_DELIMITER}Undocumented aliens were transported via Interstate 988 to avoid detection{TUPLE_DELIMITER}9)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}ILLEGAL WEAPONS{TUPLE_DELIMITER}STASH HOUSE{TUPLE_DELIMITER}Illegal weapons were stored in the stash house before being distributed{TUPLE_DELIMITER}9)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}UNDOCUMENTED ALIENS{TUPLE_DELIMITER}STASH HOUSE{TUPLE_DELIMITER}Undocumented aliens were brought to the stash house before further transport{TUPLE_DELIMITER}8)
{RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}STASH HOUSE{TUPLE_DELIMITER}VELU, GUJARAT{TUPLE_DELIMITER}The stash house was located in Velu, Gujarat serving as a hub for illegal activities{TUPLE_DELIMITER}8)
{RECORD_DELIMITER}
{COMPLETION_DELIMITER}
"""

INSTRUCTION_TEMPLATE = """
######################
-Real Data-
Below is the Real Input Data from which you have to extract Entities and Relationships as described above.
######################
Input_text: 
{input_text}
######################
Output:
"""

def create_dataset_from_output_col(file_path: str) -> Dataset:
    try:
        df = pd.read_csv(file_path).fillna('')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    processed_data = []
    print("Preparing dataset from 'Input_Text' and 'Output' columns...")
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
        print("Error: No valid data could be processed. Check CSV content and ensure 'Input_Text' and 'Output' columns exist.")
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
    instruction = INSTRUCTION_TEMPLATE.format(input_text=input_text)
    
    full_prompt = f"{SYSTEM_PROMPT.strip()}\n\n{instruction.strip()}"
    
    return f"[INST] {full_prompt} [/INST]\n"


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
            
            prompt = create_inference_prompt(
                example["input_text"]
            )

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