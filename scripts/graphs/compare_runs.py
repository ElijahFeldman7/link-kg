
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_metrics(file_path):
    metrics = {
        'entity_f1': [],
        'relationship_f1': [],
        'relationship_score_mae': []
    }
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'metrics' in data:
                metrics['entity_f1'].append(data['metrics'].get('entity_f1', 0))
                metrics['relationship_f1'].append(data['metrics'].get('relationship_f1', 0))
                metrics['relationship_score_mae'].append(data['metrics'].get('relationship_score_mae', 0))
    return metrics

def plot_comparison(baseline_metrics, finetuned_metrics, metric_name, output_dir):
    plt.figure(figsize=(10, 6))
    plt.boxplot([baseline_metrics[metric_name], finetuned_metrics[metric_name]],
                labels=['Baseline', 'Finetuned'])
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison')
    plt.ylabel('Score')
    
    mean_baseline = np.mean(baseline_metrics[metric_name])
    mean_finetuned = np.mean(finetuned_metrics[metric_name])
    plt.text(1, plt.ylim()[1]*0.9, f'Mean: {mean_baseline:.2f}', ha='center', va='center', backgroundcolor='white')
    plt.text(2, plt.ylim()[1]*0.9, f'Mean: {mean_finetuned:.2f}', ha='center', va='center', backgroundcolor='white')

    plot_filename = os.path.join(output_dir, f'{metric_name}_comparison.png')
    plt.savefig(plot_filename)
    print(f'Saved plot to {plot_filename}')
    plt.close()

def main(baseline_file, finetuned_file, output_dir):
    """Main function to load data, generate plots, and print summaries."""
    baseline_metrics = load_metrics(baseline_file)
    finetuned_metrics = load_metrics(finetuned_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metric in ['entity_f1', 'relationship_f1', 'relationship_score_mae']:
        plot_comparison(baseline_metrics, finetuned_metrics, metric, output_dir)

    print("\n--- Average Scores ---")
    print("Metric                  | Baseline | Finetuned")
    print("------------------------|----------|----------")
    for metric in ['entity_f1', 'relationship_f1', 'relationship_score_mae']:
        mean_baseline = np.mean(baseline_metrics[metric])
        mean_finetuned = np.mean(finetuned_metrics[metric])
        print(f'{metric:<23} | {mean_baseline:<8.2f} | {mean_finetuned:<8.2f}')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python compare_runs.py <baseline_metrics.jsonl> <finetuned_metrics.jsonl> <output_dir>")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    finetuned_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    main(baseline_file, finetuned_file, output_dir)
