import pandas as pd
from datasets import Dataset
from .config import MAX_LENGTH

def load_data(file_path: str) -> (Dataset, Dataset):
    try:
        df = pd.read_csv(file_path).fillna('')
        df = df.dropna(subset=['Input_Text', 'Output'])
        df = df[df['Input_Text'].str.strip() != '']
        df = df[df['Output'].str.strip() != '']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not read '{file_path}'. Make sure it exists and has 'Input_Text' and 'Output' columns. Details: {e}")
        exit()

    if len(df) == 0:
        print("Error: No valid data found in the CSV.")
        exit()

    dataset = Dataset.from_pandas(df)
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")
    return train_dataset, eval_dataset

def create_preprocess_function(tokenizer, system_prompt):
    INSTRUCTION_TEMPLATE = """Input_text: 
{input_text}
Output:
"""
    formatted_system_prompt = system_prompt.format(
        tuple_delimiter="|",
        record_delimiter="\n",
        completion_delimiter="<END>"
    )

    def preprocess_function(examples):
        inputs = [INSTRUCTION_TEMPLATE.format(input_text=text) for text in examples['Input_Text']]
        
        ground_truths = []
        for gt in examples['Output']:
            clean_gt = gt.replace("{tuple_delimiter}", "|")
            clean_gt = clean_gt.replace("{record_delimiter}", "\n")
            clean_gt = clean_gt.replace("{completion_delimiter}", "<END>")
            ground_truths.append(clean_gt)

        all_messages = [
            [
                {"role": "system", "content": formatted_system_prompt},
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
        )
        
        all_labels = []
        for i, full_text_ids in enumerate(results["input_ids"]):
            prompt_ids = tokenizer(prompt_strs[i], add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            labels = full_text_ids.copy()
            
            labels[:prompt_len] = [-100] * prompt_len
            
            if tokenizer.pad_token_id is not None:
                for j in range(len(labels)):
                    if full_text_ids[j] == tokenizer.pad_token_id:
                        labels[j] = -100

            all_labels.append(labels)

        results["labels"] = all_labels
        return results

    return preprocess_function