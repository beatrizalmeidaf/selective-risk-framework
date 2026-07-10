from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import numpy as np
import torch

def compute_accuracy_f1(preds, targets):
    """
    Calcula Acurácia global e F1-Score (Macro e Micro).
    Essas métricas medem a precisão puramente classificatória, sem considerar rejeição seletiva.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    acc = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
    f1_micro = f1_score(targets, preds, average='micro', zero_division=0)
    
    prec_macro = precision_score(targets, preds, average='macro', zero_division=0)
    rec_macro = recall_score(targets, preds, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(targets, preds)
    
    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }
