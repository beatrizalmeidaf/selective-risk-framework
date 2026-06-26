import numpy as np
import torch

def compute_selective_risk_coverage(confidences, preds, targets):
    """
    Calcula os pontos da curva Risco-Cobertura.
    Baseado em "Selective Classification for Deep Neural Networks" (Geifman & El-Yaniv, 2017).
    """
    if isinstance(confidences, torch.Tensor): confidences = confidences.cpu().numpy()
    if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()
        
    n_samples = len(confidences)
    
    # Ordenar pela confiança em ordem decrescente (mais confidente primeiro)
    sorted_idx = np.argsort(-confidences)
    sorted_preds = preds[sorted_idx]
    sorted_targets = targets[sorted_idx]
    
    # True se o modelo errou (Risco)
    errors = (sorted_preds != sorted_targets).astype(float)
    
    # Risco acumulado: erro médio sobre os exemplos aceitos até k
    cumulative_errors = np.cumsum(errors)
    k_range = np.arange(1, n_samples + 1)
    
    risks = cumulative_errors / k_range
    coverages = k_range / n_samples
    
    return risks, coverages

def compute_aurc(confidences, preds, targets):
    """
    Calcula a Área Sob a Curva de Risco-Cobertura (AURC).
    """
    risks, coverages = compute_selective_risk_coverage(confidences, preds, targets)
    # A área sob a curva usando a regra do trapézio (embora a soma simples muitas vezes seja usada no paper)
    aurc = np.trapz(risks, coverages)
    return {"aurc": aurc}

def compute_risk_at_coverage(confidences, preds, targets, target_coverages=[0.8, 0.9, 0.95]):
    """
    Retorna o Risco Seletivo (Selective Risk) para percentuais exatos de Cobertura (Coverage k).
    Geifman & El-Yaniv 2017: "Qual é a nossa taxa de erro se aceitarmos apenas as k% amostras mais confiáveis?"
    """
    risks, coverages = compute_selective_risk_coverage(confidences, preds, targets)
    
    results = {}
    for cov in target_coverages:
        # Encontrar o índice mais próximo da cobertura desejada
        idx = np.searchsorted(coverages, cov)
        if idx >= len(risks):
            idx = len(risks) - 1
        results[f"risk_at_cov_{int(cov*100)}"] = risks[idx]
        
    return results

def compute_sgr_coverage_at_risk(confidences, preds, targets, target_risks=[0.01, 0.05, 0.10]):
    """
    Usa o SGR (Selection with Guaranteed Risk) para achar qual a cobertura retida 
    para travar um risco máximo (ex: 1%, 5%, 10%).
    """
    from methods.sgr.sgr import SGRController
    sgr = SGRController(delta=0.001)
    
    results = {}
    for r_star in target_risks:
        theta, bound, coverage = sgr.fit(confidences, preds, targets, r_star)
        results[f"sgr_coverage_at_risk_{int(r_star*100)}"] = coverage
        
    return results

def evaluate_selective(confidences, preds, targets):
    """Agregador das métricas de predição seletiva."""
    metrics = compute_aurc(confidences, preds, targets)
    metrics.update(compute_risk_at_coverage(confidences, preds, targets))
    metrics.update(compute_sgr_coverage_at_risk(confidences, preds, targets))
    return metrics
