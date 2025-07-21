try:
    df = pd.read_csv("dataset.csv").dropna()
except FileNotFoundError:
    print("Error: 'dataset.csv' not found")


df = df.rename(columns={"Input_Text": "text", "Entity_Types": "entity_types","Output": "labels"})
df["labels"] = df["labels"].apply(lambda x: x.strip())
dataset = Dataset.from_pandas(df)

model_name = "Universal-NER/UniNER-7B-all"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

def preprocess(example):
    entity_types = example['entity_types']
    input_text = example['text']

    prompt = f"""-Goal-
You are an expert in Named Entity and Relationship Extraction (NER-RE) with a specialization in extracting entities and relationships from legal case documents related to human smuggling. You are highly skilled at identifying and extracting only entities of the specified entity types [{entity_types}], as well as extracting explicit relationships between them. These extracted entities and relationships will be used to build a Knowledge Graph, which will help researchers analyze human smuggling networks and identify patterns. Therefore, it is crucial to maintain strict factual accuracy and extract only what is explicitly stated in the input text, without inference or completion. You will receive entity definitions, input text, and structured examples demonstrating the correct extraction process. Study these examples carefully before performing extraction on the real input data.

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
1. Extract entities of [{entity_types}] only if they are explicitly written in the input document without inference or completion. For each extracted entity, extract the following information:
- entity_name: Name of the entity, capitalized.  Do not alter spellings or make corrections. The name should match exactly as written. For example, if 'Jaquez' is extracted as an entity then keep 'Jaquez'. Do not correct it to 'Jacquez'.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entitys attributes and activities

Do not extract any entities related to government organizations or legal proceedings, such as court, jury, government, prosecution, law enforcement, etc. These are out of scope and must be excluded if extracted entirely.

Extract each entity type separately in the following order:
- PERSON: Extract all PERSON entities. Title Handling for PERSON entities: If a person’s name appears with a title (e.g., "Border Patrol Agent Bafford Sallee", "Agent Rodriguez", or "Officer David"), extract only the person’s full name (e.g., "Bafford Sallee") as the entity_name. The title (e.g., "Border Patrol Agent") must be included in the entity_description, not in the entity_name.This prevents duplicate nodes and ensures consistent representation of individuals in the knowledge graph.
- LOCATION: Extract all LOCATION entities. If a city and a state appear together (e.g., 'Laredo, Texas' or 'Tucson, Arizona'), treat them as one LOCATION entity in the format 'City, Full State Name'. Do not split them into separate LOCATION entities.
- MEANS_OF_TRANSPORTATION: Extract all MEANS_OF_TRANSPORTATION entities.
- MEANS_OF_COMMUNICATION: Extract all MEANS_OF_COMMUNICATION entities.
- ROUTES: Extract all ROUTES entities.
- SMUGGLED_ITEMS: Extract all SMUGGLED_ITEMS entities
- ORGANIZATION: Extract all ORGANIZATION entities

Format each entity as ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other. Extract all relationships stated explicitly in the input text, even if indirect or embedded in complex structures.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: A numeric score between 0 and 10 indicating the strength of the relationship, based on the following criteria. 0 to 3 (Weak): The relationship is mentioned indirectly, with minimal context. Sentences containing "may have...", "allegedly...", or other uncertain phrasing fall into this category. 4 to 6 (Moderate): The relationship is explicitly stated but lacks detailed context, supporting evidence, or additional information. If the sentence expresses uncertainty but does not use "may have" or "allegedly," it may still fall into this range. 7 to 10 (Strong): The relationship is explicitly stated with clear, detailed context, repeated mentions, or strong supporting evidence. Sentences using direct verb tenses (e.g., "did", "was", "used", "transported") without hedging terms should be rated in this range.

 Format each relationship as ("relationship"{{tuple_delimiter}}<source_entity>{{tuple_delimiter}}<target_entity>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_strength>)

3. If any government-related entities or relationships were mistakenly extracted (e.g., court, jury, government, prosecution, law enforcement, etc.), remove them. These are out of scope for this task.

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{{record_delimiter}}** as the list delimiter.

5. When finished, output {{completion_delimiter}}


######################
-Examples-
Below are four structured examples illustrating entity and relationship extraction. Each example consists of entity types, input text, and the correct output format. Use these examples to learn the correct extraction process.
######################
Example 01:
Entity_types: PERSON, MEANS_OF_TRANSPORTATION
Input_text:
On March 12, 2024, Sai Deshpande, a known smuggler, drove an 18-wheeler carrying undocumented migrants.
######################
Output:
("entity"{{tuple_delimiter}}SAI DESHPANDE{{tuple_delimiter}}PERSON{{tuple_delimiter}}A known smuggler responsible for transporting migrants in an 18-wheeler)
{{record_delimiter}}
("entity"{{tuple_delimiter}}SMUGGLER{{tuple_delimiter}}PERSON{{tuple_delimiter}}An individual engaged in illegal human smuggling activities)
{{record_delimiter}}
("entity"{{tuple_delimiter}}18-WHEELER{{tuple_delimiter}}MEANS_OF_TRANSPORTATION{{tuple_delimiter}}A large truck used for smuggling operations)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SAI DESHPANDE{{tuple_delimiter}}SMUGGLER{{tuple_delimiter}}Sai Deshpande is identified as a smuggler involved in this case{{tuple_delimiter}}8)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SAI DESHPANDE{{tuple_delimiter}}18-WHEELER{{tuple_delimiter}}Sai Deshpande drove the 18-wheeler carrying undocumented migrants{{tuple_delimiter}}9)
{{record_delimiter}}
{{completion_delimiter}}


######################
Example 02:
Entity_types: PERSON, ROUTES, MEANS_OF_COMMUNICATION, ORGANIZATION
Input_text:
Smugglers from the Horizon Smuggling Ring used remote desert roads to avoid law enforcement, communicating via WhatsApp. The District Court later issued an order against the smuggling ring, and the Government launched an investigation.
######################
Output:
("entity"{{tuple_delimiter}}SMUGGLERS{{tuple_delimiter}}PERSON{{tuple_delimiter}}Individuals engaged in illegal human smuggling activities)
{{record_delimiter}}
("entity"{{tuple_delimiter}}HORIZON SMUGGLING RING{{tuple_delimiter}}ORGANIZATION{{tuple_delimiter}}An organized smuggling group involved in human trafficking and illegal transportation activities)
{{record_delimiter}}
("entity"{{tuple_delimiter}}REMOTE DESERT ROADS{{tuple_delimiter}}ROUTES{{tuple_delimiter}}A smuggling route used to move migrants undetected)
{{record_delimiter}}
("entity"{{tuple_delimiter}}WHATSAPP{{tuple_delimiter}}MEANS_OF_COMMUNICATION{{tuple_delimiter}}Application used by smugglers to coordinate and evade law enforcement)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SMUGGLERS{{tuple_delimiter}}HORIZON SMUGGLING RING{{tuple_delimiter}}The smugglers were associated with the Horizon Smuggling Ring{{tuple_delimiter}}7)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SMUGGLERS{{tuple_delimiter}}REMOTE DESERT ROADS{{tuple_delimiter}}Smugglers used this route to avoid law enforcement{{tuple_delimiter}}8)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}SMUGGLERS{{tuple_delimiter}}WHATSAPP{{tuple_delimiter}}Smugglers used WhatsApp to coordinate while avoiding detection{{tuple_delimiter}}7)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}HORIZON SMUGGLING RING{{tuple_delimiter}}REMOTE DESERT ROADS{{tuple_delimiter}}The Horizon Smuggling Ring used this route for illegal transportation{{tuple_delimiter}}7)
{{record_delimiter}}
{{completion_delimiter}}


######################
Example 03:
Entity_types: PERSON, SMUGGLED_ITEMS, LOCATION, ROUTES
Input_text:
Krish Patil transported undocumented aliens along Interstate 988 before arriving at a stash house in Velu, Gujarat where illegal weapons were stored.
######################
Output:
("entity"{{tuple_delimiter}}KRISH PATIL{{tuple_delimiter}}PERSON{{tuple_delimiter}}A smuggler involved in transporting undocumented aliens and illegal weapons)
{{record_delimiter}}
("entity"{{tuple_delimiter}}SMUGGLER{{tuple_delimiter}}PERSON{{tuple_delimiter}}An individual engaged in illegal human smuggling activities)
{{record_delimiter}}
("entity"{{tuple_delimiter}}UNDOCUMENTED ALIENS{{tuple_delimiter}}SMUGGLED_ITEMS{{tuple_delimiter}}A group of individuals smuggled across the border without legal documentation)
{{record_delimiter}}
("entity"{{tuple_delimiter}}ILLEGAL WEAPONS{{tuple_delimiter}}SMUGGLED_ITEMS{{tuple_delimiter}}Firearms and other restricted weapons illegally transported and stored)
{{record_delimiter}}
("entity"{{tuple_delimiter}}INTERSTATE 988{{tuple_delimiter}}ROUTES{{tuple_delimiter}}A known smuggling route used to transport undocumented aliens without detection)
{{record_delimiter}}
("entity"{{tuple_delimiter}}VELU, GUJARAT{{tuple_delimiter}}LOCATION{{tuple_delimiter}}A city where illegal weapons were stored and smuggling operations were coordinated)
{{record_delimiter}}
("entity"{{tuple_delimiter}}STASH HOUSE{{tuple_delimiter}}LOCATION{{tuple_delimiter}}A hidden facility used to shelter undocumented aliens and store illegal weapons before further transport)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}KRISH PATIL{{tuple_delimiter}}SMUGGLER{{tuple_delimiter}}Krish Patil is identified as a smuggler involved in this case{{tuple_delimiter}}9)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}KRISH PATIL{{tuple_delimiter}}UNDOCUMENTED ALIENS{{tuple_delimiter}}Krish Patil was responsible for smuggling undocumented aliens along Interstate 988{{tuple_delimiter}}10)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}KRISH PATIL{{tuple_delimiter}}ILLEGAL WEAPONS{{tuple_delimiter}}Krish Patil was involved in smuggling and storing illegal weapons at the stash house{{tuple_delimiter}}9)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}UNDOCUMENTED ALIENS{{tuple_delimiter}}INTERSTATE 988{{tuple_delimiter}}Undocumented aliens were transported via Interstate 988 to avoid detection{{tuple_delimiter}}9)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}ILLEGAL WEAPONS{{tuple_delimiter}}STASH HOUSE{{tuple_delimiter}}Illegal weapons were stored in the stash house before being distributed{{tuple_delimiter}}9)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}UNDOCUMENTED ALIENS{{tuple_delimiter}}STASH HOUSE{{tuple_delimiter}}Undocumented aliens were brought to the stash house before further transport{{tuple_delimiter}}8)
{{record_delimiter}}
("relationship"{{tuple_delimiter}}STASH HOUSE{{tuple_delimiter}}VELU, GUJARAT{{tuple_delimiter}}The stash house was located in Velu, Gujarat serving as a hub for illegal activities{{tuple_delimiter}}8)
{{record_delimiter}}
{{completion_delimiter}}



######################
-Real Data-
Below is the Real Input Data with Entity Types from which you have to extract Entities and Relationships as described above.
######################
Entity_types: {entity_types}
Input_text:
{input_text}
######################
Output:
"""
    target = example["labels"] + tokenizer.eos_token

    prompt_tokens = tokenizer(prompt, truncation=True, max_length=2048, add_special_tokens=False)
    target_tokens = tokenizer(target, truncation=True, max_length=512, add_special_tokens=False)

    max_length = 2560

    input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]
    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)

    if padding_length > 0:
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    labels = [-100] * len(prompt_tokens["input_ids"]) + target_tokens["input_ids"]
    if padding_length > 0:
        labels += [-100] * padding_length
    else:
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


tokenized_dataset = dataset.map(preprocess, remove_columns=list(dataset.features))

splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_val_splits = splits["train"].train_test_split(test_size=0.125, seed=42)
train_dataset = train_val_splits["train"]
eval_dataset = train_val_splits["test"]
test_dataset = splits["test"]

training_args = TrainingArguments(
    output_dir="./uniner_finetuned_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4, # effective batch size = 4* 8 = 32
    num_train_epochs=10,
    learning_rate=2e-5,
    optim="paged_adamw_8bit",
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    early_stopping_patience=3, 
)
def parse_model_output(output_string: str) -> set:
    tuple_delimiter = "{tuple_delimiter}"
    record_delimiter = "{record_delimiter}"

    parsed_facts = set()
    records = output_string.strip().split(record_delimiter)
    
    for record in records:
        record = record.strip()
        match = re.search(r'\((.*)\)', record)
        if not match:
            continue
        
        content = match.group(1)
        
        try:
            parts = [p.strip() for p in content.split(tuple_delimiter)]
            record_type = parts[0].lower().replace("'", "").replace('"', '')

            if record_type == "entity" and len(parts) >= 3:
                entity_name = parts[1].upper()
                entity_type = parts[2].upper()
                parsed_facts.add(('entity', entity_type, entity_name))

            elif record_type == "relationship" and len(parts) >= 3:
                source_entity = parts[1].upper()
                target_entity = parts[2].upper()
                parsed_facts.add(('relationship', source_entity, target_entity))

        except IndexError:
            continue
            
    return parsed_facts

def calculate_metrics(predicted_string: str, ground_truth_string: str) -> dict:
    predicted_facts = parse_model_output(predicted_string)
    truth_facts = parse_model_output(ground_truth_string)
    
    true_positives = len(predicted_facts.intersection(truth_facts))
    false_positives = len(predicted_facts.difference(truth_facts))
    false_negatives = len(truth_facts.difference(predicted_facts))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def compute_metrics_wrapper(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predicted_ids = np.argmax(predictions, axis=-1)

    labels[labels == -100] = tokenizer.pad_token_id
    
    predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    label_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)

    all_metrics = {"precision": [], "recall": [], "f1_score": []}

    for pred_str, label_str in zip(predicted_strings, label_strings):
        pred_str_cleaned = pred_str.strip()
        label_str_cleaned = label_str.strip()

        if not pred_str_cleaned or not label_str_cleaned:
            continue

        metrics = calculate_metrics(pred_str_cleaned, label_str_cleaned)
        all_metrics["precision"].append(metrics["precision"])
        all_metrics["recall"].append(metrics["recall"])
        all_metrics["f1_score"].append(metrics["f1_score"])

    avg_metrics = {
        "precision": np.mean(all_metrics["precision"]),
        "recall": np.mean(all_metrics["recall"]),
        "f1_score": np.mean(all_metrics["f1_score"])
    }
    
    return avg_metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_wrapper,
)


print("training UniNER")
trainer.train()
print("Training complete.")

print("\nEvaluating on the final test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)

output_dir = "./uniner_finetuned_model"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

with open(f"{output_dir}/test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

print(f"\nTraining and evaluation complete. Model and results saved to '{output_dir}'")
print("\nFinal Test Results:")
print(json.dumps(test_results, indent=2))