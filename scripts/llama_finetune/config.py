from datetime import datetime

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "datasets/dataset5.csv"
RUN_NAME = "llama_finetune"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
NEW_MODEL_DIR = f"runs/{RUN_NAME}_{timestamp}"
MAX_LENGTH = 4096

tuple_delimiter = "{tuple_delimiter}"
record_delimiter = "{record_delimiter}"
completion_delimiter = "{completion_delimiter}"
SYSTEM_PROMPT = """
-Goal-
You are an expert in Named Entity and Relationship Extraction (NER-RE). Your task is to identify and extract entities and their relationships from a given legal text.

-Entity Types-
1. PERSON: Short name or full name of a person.
2. LOCATION: Name of any geographical location.
3. ORGANIZATION: Names of companies, organized criminal groups, etc.
4. MEANS_OF_TRANSPORTATION: The mean by which someone moves from one place to another.
5. MEANS_OF_COMMUNICATION: The mean by which communication is performed.
6. ROUTES: Names of roads, freeways, highways, or other types of roads.
7. SMUGGLED_ITEMS: Any illegally transported goods.

-Instructions-
1. Extract entities of the defined types.
2. Extract explicit relationships between entities.
3. Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
4. Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
5. Do NOT extract entities related to government or legal proceedings.
6. When finished, output {completion_delimiter}.
"""