import re
import numpy as np
import torch
from functools import partial

tuple_delimiter = "|" 

def parse_for_eval(text: str) -> (set, dict, bool):
    entities = set()
    relationships = {} 

    text = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
    text = text.strip()

    d = re.escape(tuple_delimiter)

    entity_pattern = re.compile(
        r'\(\s*"entity"\s*' + d + r'\s*(.*?)\s*' + d + r'\s*(.*?)\s*' + d + r'.*?\)', 
        re.DOTALL
    )
    
    rel_pattern_with_score = re.compile(
        r'\(\s*"relationship"\s*' + d + r'\s*(.*?)\s*' + d + r'\s*(.*?)\s*' + d + r'.*?' + d + r'\s*(\d+)\s*\)', 
        re.DOTALL
    )

    try:
        for name, type_ in entity_pattern.findall(text):
            entities.add((name.strip().upper(), type_.strip().upper()))
        
        for ent1, ent2, score in rel_pattern_with_score.findall(text):
            sorted_ents = tuple(sorted([ent1.strip().upper(), ent2.strip().upper()]))
            relationships[sorted_ents] = int(score)

        has_content = len(entities) > 0 or len(relationships) > 0
        is_parsable = has_content
        
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

    def calc_f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return f1

    final_metrics['entity_f1'] = calc_f1(entity_tp, entity_fp, entity_fn)
    final_metrics['relationship_f1'] = calc_f1(rel_tp, rel_fp, rel_fn)
    final_metrics['relationship_score_mae'] = np.mean(score_errors) if score_errors else 0.0

    return final_metrics

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics_wrapper(eval_pred, tokenizer):
    predicted_ids, labels = eval_pred
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)
    label_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)

    header_token = "<|start_header_id|>assistant<|end_header_id|>"
    
    clean_preds = []
    clean_labels = []

    for pred, label in zip(predicted_strings, label_strings):
        if header_token in pred:
            pred_content = pred.split(header_token)[-1]
        else:
            pred_content = pred

        pred_content = pred_content.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        
        label_content = label.split(header_token)[-1] if header_token in label else label
        
        clean_preds.append(pred_content)
        clean_labels.append(label_content.strip())
    
    return compute_metrics(clean_preds, clean_labels)