import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def compute_auroc(id_scores, ood_scores):
    """
    Calcula AUROC. Assumimos que Scores Maiores = ID, Scores Menores = OOD.
    """
    if isinstance(id_scores, torch.Tensor): id_scores = id_scores.cpu().numpy()
    if isinstance(ood_scores, torch.Tensor): ood_scores = ood_scores.cpu().numpy()
        
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    return roc_auc_score(labels, scores)

def compute_fpr_at_tpr(id_scores, ood_scores, target_tpr=0.95):
    """
    Calcula FPR@95 TPR (FPR quando a cobertura ID é fixada em 95%).
    """
    if isinstance(id_scores, torch.Tensor): id_scores = id_scores.cpu().numpy()
    if isinstance(ood_scores, torch.Tensor): ood_scores = ood_scores.cpu().numpy()
        
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Encontrar o índice onde TPR >= target_tpr
    idx = np.searchsorted(tpr, target_tpr)
    
    if idx < len(fpr):
        return fpr[idx]
    return fpr[-1]

def compute_aupr(id_scores, ood_scores):
    """
    Calcula AUPR-IN e AUPR-OUT.
    AUPR-IN: In-distribution é a classe Positiva.
    AUPR-OUT: Out-of-distribution é a classe Positiva (scores invertidos).
    """
    if isinstance(id_scores, torch.Tensor): id_scores = id_scores.cpu().numpy()
    if isinstance(ood_scores, torch.Tensor): ood_scores = ood_scores.cpu().numpy()
        
    # AUPR-IN
    labels_in = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores_in = np.concatenate([id_scores, ood_scores])
    aupr_in = average_precision_score(labels_in, scores_in)
    
    # AUPR-OUT
    labels_out = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores_out = -np.concatenate([id_scores, ood_scores]) # Inverter scores
    aupr_out = average_precision_score(labels_out, scores_out)
    
    return {
        "aupr_in": aupr_in,
        "aupr_out": aupr_out
    }

def evaluate_ood(id_scores, ood_scores):
    """Agregador das métricas OOD."""
    if len(ood_scores) == 0:
        return {}
        
    return {
        "auroc": compute_auroc(id_scores, ood_scores),
        "fpr_at_95": compute_fpr_at_tpr(id_scores, ood_scores, target_tpr=0.95),
        **compute_aupr(id_scores, ood_scores)
    }
