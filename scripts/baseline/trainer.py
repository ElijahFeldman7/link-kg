import json
import os
import sys
import time
import getpass
import socket
from transformers import Trainer
from tqdm import tqdm
import torch
from scripts.uniner_train.llama_finetune.metrics import compute_metrics
from . import config as baseline_config


class CustomBaselineTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_run_description()

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

        if os.path.exists(summary_report_path):
            os.remove(summary_report_path)
        if os.path.exists(metrics_jsonl_path):
            os.remove(metrics_jsonl_path)

        all_metrics = {
            "eval_loss": 0.0,
            "eval_f1": 0.0,
            "eval_mae": 0.0,
            "eval_correct": 0,
            "eval_total": 0,
        }

        with open(summary_report_path, "w") as summary_writer, open(
            metrics_jsonl_path, "w"
        ) as jsonl_writer:
            for sample in tqdm(eval_dataset, desc="Evaluating"):
                prompt = sample["text"]
                ground_truth = sample["output"]

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=100)

                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                prediction_text = prediction[len(prompt) :].strip()

                metrics = compute_metrics((prediction_text, ground_truth))

                summary_writer.write(f"Prompt: {prompt}\n")
                summary_writer.write(f"Prediction: {prediction_text}\n")
                summary_writer.write(f"Ground Truth: {ground_truth}\n")
                summary_writer.write(f"Metrics: {metrics}\n")
                summary_writer.write("-" * 20 + "\n")

                metric_data = {
                    "prompt": prompt,
                    "prediction": prediction_text,
                    "ground_truth": ground_truth,
                    **metrics,
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
