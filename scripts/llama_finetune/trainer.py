import os
import json
import sys
import time
import getpass
import socket
from transformers import Trainer
from .metrics import compute_metrics
from . import config as llama_finetune_config


class CustomTrainer(Trainer):
    def __init__(self, *args, raw_eval_dataset=None, system_prompt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_eval_dataset = raw_eval_dataset
        self.system_prompt = system_prompt
        self.save_run_description()

    def save_run_description(self):
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        desc = []
        desc.append("Run Type: llama_finetune")
        desc.append(f"Start Time: {time.ctime()}")

        try:
            desc.append(f"Model ID: {self.model.name_or_path}")
        except AttributeError:
            desc.append("Model ID: Not available")
            
        desc.append(f"User: {getpass.getuser()}")
        desc.append(f"Hostname: {socket.gethostname()}")
        desc.append(f"Run Command: {' '.join(sys.argv)}")

        if self.train_dataset:
            desc.append(f"Training Dataset Size: {len(self.train_dataset)}")
        if self.eval_dataset:
            desc.append(f"Evaluation Dataset Size: {len(self.eval_dataset)}")
            
        desc.append("\n--- Training Arguments ---\n")
        desc.append(self.args.to_json_string())

        desc.append("\n--- Custom Config (from scripts.llama_finetune.config) ---\n")
        for key, value in vars(llama_finetune_config).items():
            if not key.startswith("__") and not isinstance(
                value, type(sys)
            ):
                desc.append(f"{key}: {value}")

        desc_path = os.path.join(output_dir, "desc.txt")
        with open(desc_path, "w") as f:
            f.write("\n".join(desc))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        predictions = self.predict(self.eval_dataset)
        predicted_ids = predictions.predictions
        labels = predictions.label_ids

        labels[labels == -100] = self.tokenizer.pad_token_id

        predicted_strings = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        label_strings = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        prompts = [item['Input_Text'] for item in self.raw_eval_dataset]

        assistant_header = "assistant\n\n"

        total_metrics = {
            "parsability_score": 0.0,
            "entity_f1": 0.0,
            "relationship_f1": 0.0,
            "relationship_score_mae": 0.0
        }
        num_samples = 0
        
        summary_report_parts = []
        jsonl_report = []

        if self.system_prompt:
            summary_report_parts.append(f"System Prompt:\n{self.system_prompt}\n\n")

        for i, (pred, label, prompt_text) in enumerate(zip(predicted_strings, label_strings, prompts)):
            pred_assistant_part = pred.split(assistant_header)[-1].strip()
            label_assistant_part = label.split(assistant_header)[-1].strip()

            metrics = compute_metrics([pred_assistant_part], [label_assistant_part])
            
            for key in total_metrics:
                if key in metrics:
                    total_metrics[key] += metrics[key]
            num_samples += 1

            report_part = f"--- Sample {i} ---\n"
            report_part += f"Prompt:\n{prompt_text}\n\n"
            report_part += f"Prediction:\n{pred_assistant_part}\n"
            report_part += f"Ground Truth:\n{label_assistant_part}\n"
            report_part += f"Metrics: {json.dumps(metrics, indent=2)}\n\n"
            summary_report_parts.append(report_part)

            jsonl_report.append({
                "sample_id": i,
                "prompt": prompt_text,
                "prediction": pred_assistant_part,
                "ground_truth": label_assistant_part,
                "metrics": metrics
            })
        
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
        
        summary_report = "--- Average Metrics ---\n"
        summary_report += json.dumps(avg_metrics, indent=2) + "\n\n"
        summary_report += "".join(summary_report_parts)


        summary_path = os.path.join(self.args.output_dir, "summary_report.txt")
        with open(summary_path, "w") as f:
            f.write(summary_report)

        jsonl_path = os.path.join(self.args.output_dir, "metrics.jsonl")
        with open(jsonl_path, "w") as f:
            for entry in jsonl_report:
                f.write(json.dumps(entry) + "\n")

        return eval_output
