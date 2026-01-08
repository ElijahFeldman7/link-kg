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
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
             raise ValueError("Trainer: No evaluation dataset found (eval_dataset is None).")

        predictions_output = self.predict(eval_dataset)
        
        predicted_ids = predictions_output.predictions
        tokenizer = getattr(self, "processing_class", self.tokenizer)
        predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)
        
        original_texts = [item['Input_Text'] for item in self.raw_eval_dataset]
        original_ground_truths = [item['Output'] for item in self.raw_eval_dataset]

        decoded_predictions = []
        decoded_ground_truths = []

        HEADER_PATTERN = "assistant<|end_header_id|>"

        for i in range(len(predicted_strings)):
            pred_text = predicted_strings[i]
            
            if HEADER_PATTERN in pred_text:
                pred_assistant_part = pred_text.split(HEADER_PATTERN)[-1]
            else:
                if "assistant" in pred_text:
                     pred_assistant_part = pred_text.split("assistant")[-1]
                else:
                     pred_assistant_part = pred_text

            if "<END>" in pred_assistant_part:
                pred_assistant_part = pred_assistant_part.split("<END>")[0] + "<END>"
            
            if "<|start_header_id|>" in pred_assistant_part:
                pred_assistant_part = pred_assistant_part.split("<|start_header_id|>")[0]

            pred_assistant_part = pred_assistant_part.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            
            decoded_predictions.append(pred_assistant_part)
            
            gt = original_ground_truths[i].strip()
            gt = gt.replace("{tuple_delimiter}", "|").replace("{record_delimiter}", "\n").replace("{completion_delimiter}", "<END>")
            decoded_ground_truths.append(gt)

        global_metrics = compute_metrics(decoded_predictions, decoded_ground_truths)
        
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if eval_output is None: eval_output = {}
        for key, value in global_metrics.items():
            eval_output[f"{metric_key_prefix}_{key}"] = value

        summary_report_parts = []
        jsonl_report = []

        for i, (pred, label, prompt_text) in enumerate(zip(decoded_predictions, decoded_ground_truths, original_texts)):
            
            sample_metrics = compute_metrics([pred], [label])

            report_part = f"--- Sample {i} ---\n"
            report_part += f"Input Data: {prompt_text}\n\n" 
            report_part += f"Prediction:\n{pred}\n"
            report_part += f"Ground Truth:\n{label}\n"
            report_part += f"Metrics: {json.dumps(sample_metrics, indent=2)}\n\n"
            
            summary_report_parts.append(report_part)

            jsonl_report.append({
                "sample_id": i,
                "prompt": prompt_text,
                "prediction": pred,
                "ground_truth": label,
                "metrics": sample_metrics
            })
        
        summary_report_header = "--- GLOBAL METRICS (AVERAGE) ---\n"
        summary_report_header += json.dumps(eval_output, indent=2) + "\n"
        summary_report_header += "=" * 30 + "\n\n"
        
        summary_report = summary_report_header + "".join(summary_report_parts)

        summary_path = os.path.join(self.args.output_dir, "summary_report.txt")
        with open(summary_path, "w") as f:
            f.write(summary_report)

        jsonl_path = os.path.join(self.args.output_dir, "metrics.jsonl")
        with open(jsonl_path, "w") as f:
            for entry in jsonl_report:
                f.write(json.dumps(entry) + "\n")

        return eval_output