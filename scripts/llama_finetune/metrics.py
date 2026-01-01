import re
import numpy as np
import torch
from .config import tuple_delimiter

def parse_for_eval(text: str) -> (set, dict, bool):
    entities = set()
    relationships = {} 

    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>")[0]
        
    text = text.strip()

    entity_pattern = re.compile(
        r'\("entity"' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'.*?\)', re.DOTALL
    )
    rel_pattern_with_score = re.compile(
        r'\("relationship"' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'(.*?)' + re.escape(tuple_delimiter) + r'.*?' + re.escape(tuple_delimiter) + r'(\d+)\s*\)', re.DOTALL
    )

    try:
        for name, type in entity_pattern.findall(text):
            entities.add((name.strip().upper(), type.strip().upper()))
        
        for ent1, ent2, score in rel_pattern_with_score.findall(text):
            sorted_ents = tuple(sorted([ent1.strip().upper(), ent2.strip().upper()]))
            relationships[sorted_ents] = int(score)

        is_parsable = len(text.strip()) == 0 or (len(entities) > 0 or len(relationships) > 0)
        return entities, relationships, is_parsable
    except Exception:
        return set(), {}, False

def compute_metrics(predictions: list, ground_truths: list):
    parsability_scores, score_errors = [], []
    entity_tp, entity_fp, entity_fn = 0, 0, 0
    rel_tp, rel_fp, rel_fn = 0, 0, 0

    for pred_str, label_str in zip(predictions, ground_truths):
        pred_entities, pred_rels, is_parsable = parse_for_eval(pred_str)
        label_entities, label_rels, _ = parse_for_eval(label_str)

        parsability_scores.append(1 if is_parsable else 0)

        if not is_parsable:
            entity_fn += len(label_entities)
            rel_fn += len(label_rels)
            continue

        entity_tp += len(pred_entities & label_entities)
        entity_fp += len(pred_entities - label_entities)
        entity_fn += len(label_entities - pred_entities)

        pred_rel_keys = set(pred_rels.keys())
        label_rel_keys = set(label_rels.keys())
        rel_tp += len(pred_rel_keys & label_rel_keys)
        rel_fp += len(pred_rel_keys - label_rel_keys)
        rel_fn += len(label_rel_keys - pred_rel_keys)
        
        true_positive_rels = pred_rel_keys & label_rel_keys
        for rel_pair in true_positive_rels:
            if rel_pair in pred_rels and rel_pair in label_rels:
                error = abs(pred_rels[rel_pair] - label_rels[rel_pair])
                score_errors.append(error)

    final_metrics = {}
    final_metrics["parsability_score"] = np.mean(parsability_scores) if parsability_scores else 0.0

    entity_precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0.0
    entity_recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0.0
    final_metrics['entity_f1'] = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
    
    rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0.0
    rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0.0
    final_metrics['relationship_f1'] = 2 * (rel_precision * rel_recall) / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0
    
    final_metrics['relationship_score_mae'] = np.mean(score_errors) if score_errors else 0.0

    return final_metrics

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics_wrapper(eval_pred, tokenizer):
    predicted_ids, labels = eval_pred
    
    labels[labels == -100] = tokenizer.pad_token_id
    
    predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    label_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)

    assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
    
    clean_preds = []
    clean_labels = []

    for pred, label in zip(predicted_strings, label_strings):
        pred_assistant_part = pred.split(assistant_header)[-1].strip()
        label_assistant_part = label.split(assistant_header)[-1].strip()
        
        if pred_assistant_part.startswith("assistant("):
            pred_assistant_part = pred_assistant_part[len("assistant("):-1]
        
        clean_preds.append(pred_assistant_part)
        clean_labels.append(label_assistant_part)
    
    return compute_metrics(clean_preds, clean_labels)