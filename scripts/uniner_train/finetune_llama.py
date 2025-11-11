import pandas as pd
import torch
import numpy as np
import json
import re
import os
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "dataset5.csv"
NEW_MODEL_DIR = "./llama31_8b_instruct_finetuned"
MAX_LENGTH = 2048 

tuple_delimiter = "{tuple_delimiter}"
record_delimiter = "{record_delimiter}"
completion_delimiter = "{completion_delimiter}"

SYSTEM_PROMPT = """
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
- entity_name: Name of the entity, capitalized. Do not alter spellings or make corrections. The name should match exactly as written. For example, if 'Jaquez' is extracted as an entity then keep 'Jaquea'. Do not correct it to 'Jacquez'. 
- entity_type: One of the 7 defined entity types.
- entity_description: Comprehensive description of the entitys attributes and activities

Do not extract any entities related to government organizations or legal proceedings, such as court, jury, government, law enforcement, prosecution, homeland security, etc. These are out of scope and must be excluded if extracted entirely.

Extract each entity type separately in the following order:
- PERSON: Extract all PERSON entities. Title Handling for PERSON entities: If a person’s name appears with a title (e.g., "Border Patrol Agent Bafford Sallee", "Agent Rodriguez", or "Officer David"), extract only the person’s full name (e.g., "Bafford Sallee") as the entity_name. The title (e.g., "Border Patrol Agent") must be included in the entity_description, not in the entity_name.This prevents duplicate nodes and ensures consistent representation of individuals in the knowledge graph.
- LOCATION: Extract all LOCATION entities. If a city and state appear together (e.g., 'Laredo, Texas' or 'Tucson, Arizona'), treat them as one LOCATION entity in the format 'City, Full State Name'. Do not split them into separate LOCATION entities.
- MEANS_OF_TRANSPORTATION: Extract all MEANS_OF_TRANSPORTATION entities.
- MEANS_OF_COMMUNICATION: Extract all MEANS_OF_COMMUNICATION entities.
- ROUTES: Extract all ROUTES entities.
- SMUGGLED_ITEMS: Extract all SMUGGLED_ITEMS entities
- ORGANIZATION: Extract all ORGANIZATION entities

Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other. Extract all relationships stated explicitly in the input text, even if indirect or embedded in complex structures.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: A numeric score between 0 and 10 indicating the strength of the relationship, based on the following criteria. 0 to 3 (Weak): The relationship is mentioned indirectly, with minimal context. Sentences containing "may have...", "allegedly...", or other uncertain phrasing fall into this category. 4 to 6 (Moderate): The relationship is explicitly stated but lacks detailed context, supporting evidence, or additional information. If the sentence expresses uncertainty but does not use "may have" or "allegally," it may still fall into this range. 7 to 10 (Strong): The relationship is explicitly stated with clear, detailed context, repeated mentions, or strong supporting evidence. Sentences using direct verb tenses (e.g., "did", "was", "used", "transported") without hedging terms should be rated in this range.

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. If any government-related entities or relationships were mistakenly extracted (e.g., court, jury, government, prosecution, law enforcement, etc.), remove them. These are out of scope for this task.

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.
 
5. When finished, output {completion_delimiter}


######################
-Examples-
Below are four structured examples illustrating entity and relationship extraction. Each example consists of input text and the correct output format. Use these examples to learn the correct extraction process.
######################
Example 01:
Input_text:
On March 12, 2024, Sai Deshpande, a known smuggler, drove an 18-wheeler carrying undocumented migrants.
######################
Output:
("entity"{tuple_delimiter}SAI DESHPANDE{tuple_delimiter}PERSON{tuple_delimiter}A known smuggler responsible for transporting migrants in an 18-wheeler)
{record_delimiter}
("entity"{tuple_delimiter}SMUGGLER{tuple_delimiter}PERSON{tuple_delimiter}An individual engaged in illegal human smuggling activities)
{record_delimiter}
("entity"{tuple_delimiter}18-WHEELER{tuple_delimiter}MEANS_OF_TRANSPORTATION{tuple_delimiter}A large truck used for smuggling operations)
{record_delimiter}
("relationship"{tuple_delimiter}SAI DESHPANDE{tuple_delimiter}SMUGGLER{tuple_delimiter}Sai Deshpande is identified as a smuggler involved in this case{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}SAI DESHPANDE{tuple_delimiter}18-WHEELER{tuple_delimiter}Sai Deshpande drove the 18-wheeler carrying undocumented migrants{tuple_delimiter}9)
{record_delimiter}
{completion_delimiter}


######################
Example 02:
Input_text:
Smugglers from the Horizon Smuggling Ring used remote desert roads to avoid law enforcement, communicating via WhatsApp. The District Court later issued an order against the smuggling ring, and the Government launched an investigation.
######################
Output:
("entity"{tuple_delimiter}SMUGGLERS{tuple_delimiter}PERSON{tuple_delimiter}Individuals engaged in illegal human smuggling activities)
{record_delimiter}
("entity"{tuple_delimiter}HORIZON SMUGGLING RING{tuple_delimiter}ORGANIZATION{tuple_delimiter}An organized smuggling group involved in human trafficking and illegal transportation activities)
{record_delimiter}
("entity"{tuple_delimiter}REMOTE DESERT ROADS{tuple_delimiter}ROUTES{tuple_delimiter}A smuggling route to move migrants undetected)
{record_delimiter}
("entity"{tuple_delimiter}WHATSAPP{tuple_delimiter}MEANS_OF_COMMUNICATION{tuple_delimiter}Application used by smugglers to coordinate and evade law enforcement)
{record_delimiter}
("relationship"{tuple_delimiter}SMUGGLERS{tuple_delimiter}HORIZON SMUGGLING RING{tuple_delimiter}The smugglers were associated with the Horizon Smuggling Ring{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}SMUGGLERS{tuple_delimiter}REMOTE DESERT ROADS{tuple_delimiter}Smugglers used this route to avoid law enforcement{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}SMUGGLERS{tuple_delimiter}WHATSAPP{tuple_delimiter}Smugglers used WhatsApp to coordinate while avoiding detection{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}HORIZON SMUGGLING RING{tuple_delimiter}REMOTE DESERT ROADS{tuple_delimiter}The Horizon Smuggling Ring used this route for illegal transportation{TUPLE_DELIMITTER}7)
{record_delimiter}
{completion_delimiter}


######################
Example 03:
Input_text:
Krish Patil transported undocumented aliens along Interstate 988 before arriving at a stash house in Velu, Gujarat where illegal weapons were stored.
######################
Output:
("entity"{tuple_delimiter}KRISH PATIL{tuple_delimiter}PERSON{tuple_delimiter}A smuggler involved in transporting undocumented aliens and illegal weapons)
{record_delimiter}
("entity"{tuple_delimiter}SMUGGLER{tuple_delimiter}PERSON{tuple_delimiter}An individual engaged in illegal human smuggling activities)
{record_delimiter}
("entity"{tuple_delimiter}UNDOCUMENTED ALIENS{tuple_delimiter}SMUGGLED_ITEMS{tuple_delimiter}A group of individuals smuggled across the border without legal documentation)
{record_delimiter}
("entity"{tuple_delimiter}ILLEGAL WEAPONS{tuple_delimiter}SMUGGLED_ITEMS{tuple_delimiter}Firearms and other restricted weapons illegally transported and stored)
{record_delimiter}
("entity"{tuple_delimiter}INTERSTATE 988{tuple_delimiter}ROUTES{tuple_delimiter}A known smuggling route used to transport undocumented aliens without detection)
{record_delimiter}
("entity"{tuple_delimiter}VELU, GUJARAT{tuple_delimiter}LOCATION{tuple_delimiter}A city where illegal weapons were stored and smuggling operations were coordinated)
{record_delimiter}
("entity"{tuple_delimiter}STASH HOUSE{tuple_delimiter}LOCATION{tuple_delimiter}A hidden facility used to shelter undocumented aliens and store illegal weapons before further transport)
{record_delimiter}
("relationship"{tuple_delimiter}KRISH PATIL{tuple_delimiter}SMUGGLER{tuple_delimiter}Krish Patil is identified as a smuggler involved in this case{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}KRISH PATIL{tuple_delimiter}UNDOCUMENTED ALIENS{tuple_delimiter}Krish Patil was responsible for smuggling undocumented aliens along Interstate 988{tuple_delimiter}10)
{record_delimiter}
("relationship"{tuple_delimiter}KRISH PATIL{tuple_delimiter}ILLEGAL WEAPONS{tuple_delimiter}Krish Patil was involved in smuggling and storing illegal weapons at the stash house{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}UNDOCUMENTED ALIENS{tuple_delimiter}INTERSTATE 988{tuple_delimiter}Undocumented aliens were transported via Interstate 988 to avoid detection{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}ILLEGAL WEAPONS{tuple_delimiter}STASH HOUSE{tuple_delimiter}Illegal weapons were stored in the stash house before being distributed{tuple_delimiter}9)
{record_delimiter}
("relationship"{tuple_delimiter}UNDOCUMENTED ALIENS{tuple_delimiter}STASH HOUSE{tuple_delimiter}Undocumented aliens were brought to the stash house before further transport{tuple_delimiter}8)
{record_delimiter}
("relationship"{tuple_delimiter}STASH HOUSE{tuple_delimiter}VELU, GUJARAT{tuple_delimiter}The stash house was located in Velu, Gujarat serving as a hub for illegal activities{tuple_delimiter}8)
{record_delimiter}
{completion_delimiter}
"""

def setup_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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
    return model, tokenizer

def setup_peft_model(model):
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05, 
        bias="none", 
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_data(file_path: str) -> (Dataset, Dataset):
    try:
        df = pd.read_csv(file_path).fillna('')
        df = df.dropna(subset=['Input_Text', 'Output'])
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not read '{file_path}'. Make sure it exists and has 'Input_Text' and 'Output' columns. Details: {e}")
        exit()

    if len(df) == 0:
        print("Error: No valid data found in the CSV.")
        exit()

    dataset = Dataset.from_pandas(df)
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")
    return train_dataset, eval_dataset

def create_preprocess_function(tokenizer, system_prompt):
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
    ent_pattern = re.compile(
        r'(\("entity"' + re.escape(tuple_delimiter) + r'.*?' + re.escape(tuple_delimiter) + r'.*?' + re.escape(tuple_delimiter) + r')(.*?)(\))', 
        re.DOTALL
    )
    rel_pattern = re.compile(
        r'(\("relationship"' + re.escape(tuple_delimiter) + r'.*?' + re.escape(tuple_delimiter) + r'.*?' + re.escape(tuple_delimiter) + r')(.*?)(' + re.escape(tuple_delimiter) + r'\d+\s*\))', 
        re.DOTALL
    )
    def preprocess_function(examples):
        inputs = [INSTRUCTION_TEMPLATE.format(input_text=text) for text in examples['Input_Text']]
        ground_truths = examples['Output']
        all_messages = [
            [
                {"role": "system", "content": system_prompt.strip()},

                {"role": "user", "content": input_content.strip()}
            ] for input_content in inputs
        ]
        prompt_strs = [
            tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            ) for messages in all_messages
        ]
        full_texts = [f"{prompt}{gt}{tokenizer.eos_token}" for prompt, gt in zip(prompt_strs, ground_truths)]
        results = tokenizer(
            full_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_offsets_mapping=True
        )
        all_labels = []
        for i in range(len(full_texts)):
            labels = results["input_ids"][i].copy()
            prompt_char_len = len(prompt_strs[i])
            ground_truth = ground_truths[i]
            prompt_token_len = 0
            for token_idx, (start_char, end_char) in enumerate(results["offset_mapping"][i]):
                if start_char >= prompt_char_len:
                    prompt_token_len = token_idx
                    break
            else:
                prompt_token_len = len(labels)
            labels[:prompt_token_len] = [-100] * prompt_token_len
            desc_char_spans = []
            for match in ent_pattern.finditer(ground_truth):
                desc_char_spans.append(match.span(2))
            for match in rel_pattern.finditer(ground_truth):
                desc_char_spans.append(match.span(2))
            for start_char, end_char in desc_char_spans:
                full_start_char = prompt_char_len + start_char
                full_end_char = prompt_char_len + end_char
                token_start = results.char_to_token(i, full_start_char)
                token_end = results.char_to_token(i, full_end_char - 1)
                if token_start is not None and token_end is not None:
                    mask_len = (token_end - token_start) + 1
                    labels[token_start : token_end + 1] = [-100] * mask_len
            all_labels.append(labels)
        results["labels"] = all_labels
        del results["offset_mapping"]
        return results
    return preprocess_function

def parse_for_eval(text: str) -> (set, dict, bool):
    entities = set()
    relationships = {} 

    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>")[0]
        
    text = text.strip()

    entity_pattern = re.compile(
        r'\("entity"' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'.*?\)', re.DOTALL
    )
    rel_pattern_with_score = re.compile(
        r'\("relationship"' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'.*?' + re.escape(tuple_delimiter) + r'(\d+)\s*\)', re.DOTALL
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

def compute_metrics_wrapper(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_ids = np.argmax(predictions, axis=-1)
    
    labels[labels == -100] = tokenizer.pad_token_id
    
    predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    label_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)

    assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
    
    clean_preds = []
    clean_labels = []

    for pred, label in zip(predicted_strings, label_strings):
        pred_assistant_part = pred.split(assistant_header)[-1].strip()
        label_assistant_part = label.split(assistant_header)[-1].strip()
        
        clean_preds.append(pred_assistant_part)
        clean_labels.append(label_assistant_part)
    
    return compute_metrics(clean_preds, clean_labels)


if __name__ == "__main__":
    model, tokenizer = setup_model_and_tokenizer()
    
    model = setup_peft_model(model)
    
    train_dataset, eval_dataset = load_data(DATASET_PATH)
    
    preprocess_function = create_preprocess_function(tokenizer, SYSTEM_PROMPT)
    
    print("Tokenizing train dataset...")
    tokenized_train = train_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=list(train_dataset.features)
    )
    print("Tokenizing evaluation dataset...")
    tokenized_eval = eval_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=list(eval_dataset.features)
    )

    training_args = TrainingArguments(
        output_dir=NEW_MODEL_DIR,
        per_device_train_batch_size=1,      
        gradient_accumulation_steps=16,     
        per_device_eval_batch_size=1,
        num_train_epochs=5,                
        learning_rate=2e-4,                
        optim="paged_adamw_8bit",
        bf16=True,                         
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_entity_f1", 
        greater_is_better=True,
        report_to="tensorboard",           
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
    )

    print("\n--- Starting Llama 3.1 QLoRA Fine-tuning ---")
    trainer.train()
    print("\n--- Training Complete ---")

    best_model_path = os.path.join(NEW_MODEL_DIR, "best_model")
    trainer.save_model(best_model_path)
    print(f"Best model saved to {best_model_path}")