from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import numpy as np
import torch

def compute_accuracy_f1(preds, targets):
    """
    Calcula Acurácia global e F1-Score (Macro e Micro).
    Essas métricas medem a precisão puramente classificatória, sem considerar rejeição seletiva.

    IMPORTANTE: quando 'targets' contém exemplos OOD (label -1), eles nunca podem
    ser acertados por um classificador fechado, então 'accuracy' fica estruturalmente
    limitada a (1 - ood_fraction), mesmo com um classificador perfeito nas classes ID.

    Por isso 'id_only_accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
    'f1_micro' e 'balanced_accuracy' são calculadas SÓ no subconjunto ID: se
    incluíssem OOD, o sklearn trataria a label -1 como uma classe fantasma (nunca
    predita) com precision/recall/F1 = 0, derrubando essas médias mesmo com um
    classificador perfeito nas classes ID. 'accuracy' e 'ood_fraction' continuam
    sendo calculadas no conjunto completo (ID+OOD) de propósito, como teto teórico.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    acc = accuracy_score(targets, preds)

    id_mask = targets != -1
    ood_fraction = float((~id_mask).sum() / len(targets))
    id_preds = preds[id_mask]
    id_targets = targets[id_mask]

    if id_mask.any():
        id_only_accuracy = accuracy_score(id_targets, id_preds)
        f1_macro = f1_score(id_targets, id_preds, average='macro', zero_division=0)
        f1_micro = f1_score(id_targets, id_preds, average='micro', zero_division=0)
        prec_macro = precision_score(id_targets, id_preds, average='macro', zero_division=0)
        rec_macro = recall_score(id_targets, id_preds, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(id_targets, id_preds)
    else:
        id_only_accuracy = acc
        f1_macro = f1_micro = prec_macro = rec_macro = balanced_acc = 0.0

    return {
        "accuracy": acc,
        "id_only_accuracy": id_only_accuracy,
        "ood_fraction": ood_fraction,
        "balanced_accuracy": balanced_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }
