import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
import time

MODEL_PATH = "/scratch/efeldma5/uniner_project/uniner_person_baseline" 
INPUT_CSV_PATH = "dataset5.csv"    
OUTPUT_CSV_PATH = "predictions_output.csv" 

INPUT_COLUMN_NAME = "Input"
PREDICTION_COLUMN_NAME = "Model_Prediction"

def predict_entities(text, model, tokenizer):
    instruction = "From the text provided, extract all PERSON entities. Your output must be a JSON-formatted list of strings."
    prompt = f"[INST] {instruction}\n\nText: {text} [/INST]\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False     
        )
    
    input_length = inputs.input_ids.shape[1]
    decoded_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return decoded_output.strip()


if __name__ == "__main__":
    print(f"Loading fine-tuned model from '{MODEL_PATH}'...")
    start_time = time.time()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Universal-NER/UniNER-7B-all",
        quantization_config=bnb_config,
        device_map="auto", 
    )
    
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval() 
    end_time = time.time()
    print(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")

    print(f"\nLoading input data from '{INPUT_CSV_PATH}'...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        if INPUT_COLUMN_NAME not in df.columns:
             print(f"Error: Input column '{INPUT_COLUMN_NAME}' not found in the CSV.")
             exit()
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_PATH}' was not found.")
        exit()

    print(f"Running predictions on {len(df)} rows...")
    predictions = []
    
    start_time = time.time()
    for index, row in df.iterrows():
        text_to_predict = row[INPUT_COLUMN_NAME]
        
        if not isinstance(text_to_predict, str) or not text_to_predict.strip():
            predictions.append("") 
            continue

        try:
            prediction = predict_entities(text_to_predict, model, tokenizer)
            predictions.append(prediction)
        except Exception as e:
            print(f"An error occurred on row {index}: {e}")
            predictions.append(f"ERROR: {e}")

        if (index + 1) % 10 == 0:
            print(f"  Processed {index + 1}/{len(df)} rows...")
            
    end_time = time.time()
    print(f"Predictions completed in {end_time - start_time:.2f} seconds.")

    print(f"\nSaving results to '{OUTPUT_CSV_PATH}'...")
    df[PREDICTION_COLUMN_NAME] = predictions
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n✅ Script finished successfully.")