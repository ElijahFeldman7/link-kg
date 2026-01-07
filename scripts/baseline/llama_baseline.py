import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from scripts.baseline.config import DATASET_PATH, BASE_MODEL_NAME, NEW_MODEL_DIR
from scripts.baseline.trainer import CustomBaselineTrainer
from scripts.llama_finetune.metrics import compute_metrics, parse_for_eval

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
- entity_name: Name of the entity, capitalized. Do not alter spellings or make corrections. The name should match exactly as written. For example, if 'Jaquez' is extracted as an entity then keep 'Jaquez'. Do not correct it to 'Jacquez'. 
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
- relationship_strength: A numeric score between 0 and 10 indicating the strength of the relationship, based on the following criteria. 0 to 3 (Weak): The relationship is mentioned indirectly, with minimal context. Sentences containing "may have...", "allegedly...", or other uncertain phrasing fall into this category. 4 to 6 (Moderate): The relationship is explicitly stated but lacks detailed context, supporting evidence, or additional information. If the sentence expresses uncertainty but does not use "may have" or "allegedly," it may still fall into this range. 7 to 10 (Strong): The relationship is explicitly stated with clear, detailed context, repeated mentions, or strong supporting evidence. Sentences using direct verb tenses (e.g., "did", "was", "used", "transported") without hedging terms should be rated in this range.

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. If any government-related entities or relationships were mistakenly extracted (e.g., court, jury, government, prosecution, law enforcement, etc.), remove them. These are out of scope for this task.

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.
 
5. When finished, output {completion_delimiter}
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

def create_dataset_from_output_col(file_path: str, tokenizer) -> Dataset:
    try:
        df = pd.read_csv(file_path).fillna('')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()
    tdem = '|'
    rdem = '\n'
    cdem = '<END>'
    formatted_system_prompt = SYSTEM_PROMPT.format(
        tuple_delimiter=tdem,
        record_delimiter=rdem,
        completion_delimiter=cdem
    )
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
        clean_target = clean_target.replace("{tuple_delimiter}", "|")
        clean_target = clean_target.replace("{record_delimiter}", "\n")
        clean_target = clean_target.replace("{completion_delimiter}", "<END>")

        user_content = INSTRUCTION_TEMPLATE.format(input_text=input_text)
        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": user_content.strip()}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        processed_data.append({
            "text": prompt,
            "output": clean_target,
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
    print("NOTE: This requires you to be logged in via `huggingface-cli login`.")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model.eval()
    return model, tokenizer

def main():
    model, tokenizer = setup_model_and_tokenizer()
    eval_dataset = create_dataset_from_output_col(DATASET_PATH, tokenizer)

    training_args = TrainingArguments(
        output_dir=NEW_MODEL_DIR,
        per_device_eval_batch_size=1,
        logging_dir=f"{NEW_MODEL_DIR}/logs",
    )

    trainer = CustomBaselineTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        eval_dataset=eval_dataset,
        system_prompt=formatted_system_prompt
    )

    print("\nRunning inference on the dataset...")
    trainer.evaluate()
    print(f"Metrics and summary report saved in {NEW_MODEL_DIR}")

if __name__ == "__main__":
    main()