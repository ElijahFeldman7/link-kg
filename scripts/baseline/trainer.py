import json
import os
import sys
import time
import getpass
import socket
from tqdm import tqdm
import torch
from scripts.llama_finetune.metrics import compute_metrics
from . import config as baseline_config


class CustomBaselineTrainer:
    def __init__(self, model, tokenizer, args, eval_dataset=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.eval_dataset = eval_dataset
        self.save_run_description()

    def log(self, metrics):
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    def save_run_description(self):
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        desc = []
        desc.append("Run Type: baseline")
        desc.append(f"Start Time: {time.ctime()}")
        
        try:
            desc.append(f"Model ID: {self.model.name_or_path}")
        except AttributeError:
            desc.append("Model ID: Not available")
            
        desc.append(f"User: {getpass.getuser()}")
        desc.append(f"Hostname: {socket.gethostname()}")
        desc.append(f"Run Command: {' '.join(sys.argv)}")
        
        if self.eval_dataset:
            desc.append(f"Evaluation Dataset Size: {len(self.eval_dataset)}")

        desc.append("\n--- Training Arguments ---\n")
        desc.append(self.args.to_json_string())

        desc.append("\n--- Custom Config (from scripts.baseline.config) ---\n")
        for key, value in vars(baseline_config).items():
            if not key.startswith("__") and not isinstance(
                value, type(sys)
            ):
                desc.append(f"{key}: {value}")

        desc_path = os.path.join(output_dir, "desc.txt")
        with open(desc_path, "w") as f:
            f.write("\n".join(desc))

def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        output_dir = self.args.output_dir
        summary_report_path = os.path.join(output_dir, "summary_report.txt")
        metrics_jsonl_path = os.path.join(output_dir, "metrics.jsonl")

        # Clean up old files
        if os.path.exists(summary_report_path):
            os.remove(summary_report_path)
        if os.path.exists(metrics_jsonl_path):
            os.remove(metrics_jsonl_path)

        total_metrics = {
            "parsability_score": 0.0,
            "entity_f1": 0.0,
            "relationship_f1": 0.0,
            "relationship_score_mae": 0.0
        }
        num_samples = 0

        print(f"Generating responses and saving to {summary_report_path}...")
        
        with open(summary_report_path, "w") as summary_writer, open(metrics_jsonl_path, "w") as jsonl_writer:
            
            for sample in tqdm(eval_dataset, desc="Evaluating"):
                prompt = sample["text"]
                ground_truth = sample["output"]
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                input_length = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=2048, 
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                generated_tokens = outputs[0][input_length:]
                prediction_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                metrics = compute_metrics([prediction_text], [ground_truth])

                summary_writer.write(f"Prompt: {prompt}\n")
                summary_writer.write(f"Prediction: {prediction_text}\n")
                summary_writer.write(f"Ground Truth: {ground_truth}\n")
                summary_writer.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
                summary_writer.write("-" * 20 + "\n")

                metric_data = {
                    "prompt": prompt,
                    "prediction": prediction_text,
                    "ground_truth": ground_truth,
                    "metrics": metrics,
                }
                jsonl_writer.write(json.dumps(metric_data) + "\n")
    
                for key in total_metrics:
                    if key in metrics and isinstance(metrics[key], (int, float)):
                        total_metrics[key] += metrics[key]

                num_samples += 1

        avg_metrics = {f"eval_{key}": value / num_samples for key, value in total_metrics.items() if num_samples > 0}
        
        with open(summary_report_path, "r") as f:
            existing_content = f.read()
            
        header = "--- GLOBAL METRICS (AVERAGE) ---\n"
        header += json.dumps(avg_metrics, indent=2) + "\n"
        header += "=" * 30 + "\n\n"
        
        with open(summary_report_path, "w") as f:
            f.write(header + existing_content)
            
        self.log(avg_metrics)
        return avg_metrics