import json
import os
from transformers import Trainer
from tqdm import tqdm
import torch
from scripts.uniner_train.llama_finetune.metrics import compute_metrics
from scripts.baseline.config import NEW_MODEL_DIR

class CustomBaselineTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        
        summary_report_path = os.path.join(NEW_MODEL_DIR, "summary_report.txt")
        metrics_jsonl_path = os.path.join(NEW_MODEL_DIR, "metrics.jsonl")

        # Clear previous run files if they exist
        if os.path.exists(summary_report_path):
            os.remove(summary_report_path)
        if os.path.exists(metrics_jsonl_path):
            os.remove(metrics_jsonl_path)
        
        all_metrics = {"eval_loss": 0.0, "eval_f1": 0.0, "eval_mae": 0.0, "eval_correct": 0, "eval_total": 0}
        
        with open(summary_report_path, "w") as summary_writer, open(metrics_jsonl_path, "w") as jsonl_writer:
            for sample in tqdm(eval_dataset, desc="Evaluating"):
                prompt = sample['text']
                ground_truth = sample['output']

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=100)
                
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # The prediction will contain the input prompt, so we need to remove it.
                prediction_text = prediction[len(prompt):].strip()

                metrics = compute_metrics((prediction_text, ground_truth))

                # Write to summary report
                summary_writer.write(f"Prompt: {prompt}\n")
                summary_writer.write(f"Prediction: {prediction_text}\n")
                summary_writer.write(f"Ground Truth: {ground_truth}\n")
                summary_writer.write(f"Metrics: {metrics}\n")
                summary_writer.write("-" * 20 + "\n")

                # Write to jsonl
                metric_data = {
                    "prompt": prompt,
                    "prediction": prediction_text,
                    "ground_truth": ground_truth,
                    **metrics
                }
                jsonl_writer.write(json.dumps(metric_data) + "\n")

                all_metrics["eval_f1"] += metrics["f1"]
                all_metrics["eval_mae"] += metrics["mae"]
                all_metrics["eval_correct"] += metrics["correct"]
                all_metrics["eval_total"] += 1

        num_samples = len(eval_dataset)
        all_metrics["eval_f1"] /= num_samples
        all_metrics["eval_mae"] /= num_samples
        
        self.log(all_metrics)
        
        return all_metrics
