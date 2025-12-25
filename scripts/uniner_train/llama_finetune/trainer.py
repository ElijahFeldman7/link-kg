import os
import json
from transformers import Trainer
from .metrics import compute_metrics

class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        predictions = self.predict(self.eval_dataset)
        predicted_ids = predictions.predictions
        labels = predictions.label_ids

        labels[labels == -100] = self.tokenizer.pad_token_id

        predicted_strings = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        label_strings = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

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

        for i, (pred, label) in enumerate(zip(predicted_strings, label_strings)):
            pred_assistant_part = pred.split(assistant_header)[-1].strip()
            label_assistant_part = label.split(assistant_header)[-1].strip()

            metrics = compute_metrics([pred_assistant_part], [label_assistant_part])
            
            for key in total_metrics:
                if key in metrics:
                    total_metrics[key] += metrics[key]
            num_samples += 1

            report_part = f"--- Sample {i} ---\n"
            report_part += f"Prediction:\n{pred_assistant_part}\n"
            report_part += f"Ground Truth:\n{label_assistant_part}\n"
            report_part += f"Metrics: {json.dumps(metrics, indent=2)}\n\n"
            summary_report_parts.append(report_part)

            jsonl_report.append({
                "sample_id": i,
                "prediction": pred_assistant_part,
                "ground_truth": label_assistant_part,
                "metrics": metrics
            })
        
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
        
        summary_report = "--- Average Metrics ---\n"
        summary_report += json.dumps(avg_metrics, indent=2) + "\n\n"
        summary_report += "".join(summary_report_parts)


        # Save reports
        summary_path = os.path.join(self.args.output_dir, "summary_report.txt")
        with open(summary_path, "w") as f:
            f.write(summary_report)

        jsonl_path = os.path.join(self.args.output_dir, "metrics.jsonl")
        with open(jsonl_path, "w") as f:
            for entry in jsonl_report:
                f.write(json.dumps(entry) + "\n")

        return eval_output
