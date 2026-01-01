import pandas as pd
from datasets import Dataset
from .config import MAX_LENGTH

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
    INSTRUCTION_TEMPLATE = """Input_text: 
{input_text}
Output:
"""

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
            
            prompt_token_len = 0
            for token_idx, (start_char, end_char) in enumerate(results["offset_mapping"][i]):
                if start_char >= prompt_char_len:
                    prompt_token_len = token_idx
                    break
            else:
                prompt_token_len = len(labels)

            labels[:prompt_token_len] = [-100] * prompt_token_len
            all_labels.append(labels)

        results["labels"] = all_labels
        del results["offset_mapping"]
        
        return results

    return preprocess_function