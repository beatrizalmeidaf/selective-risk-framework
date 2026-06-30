import numpy as np
import torch

def compute_ece(confidences, preds, targets, n_bins=10):
    """
    Calcula o Expected Calibration Error (ECE).
    Quantifica a discrepância entre a confiança predita (probabilidade) e a precisão empírica real.
    """
    if isinstance(confidences, torch.Tensor): confidences = confidences.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor): preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()
        
    # Normalização Min-Max se as confianças não forem probabilidades (ex: distâncias do LAQDA)
    if len(confidences) > 0 and (confidences.min() < 0 or confidences.max() > 1):
        c_min, c_max = confidences.min(), confidences.max()
        if c_max > c_min:
            confidences = (confidences - c_min) / (c_max - c_min)
        else:
            confidences = np.ones_like(confidences)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    accuracies = (preds == targets)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return {"ece": ece}
