import numpy as np
import torch

def compute_selective_risk_coverage(confidences, preds, targets):
    """
    Calcula os pontos da curva Risco-Cobertura.
    Baseado em "Selective Classification for Deep Neural Networks" (Geifman & El-Yaniv, 2017).
    """
    # Assumimos que os tensores já foram convertidos para numpy no evaluate_selective
        
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
    
    # E-AURC (Empirical AURC, normalizado pela taxa de erro base - chute aleatório ou precisão geral)
    # A métrica E-AURC penaliza modelos que começam com um erro base muito alto.
    # E-AURC = AURC - (risco_base / 2) -> (aproximação para comparação direta)
    error_rate = 1.0 - np.mean(preds == targets)
    e_aurc = aurc - (error_rate / 2.0)
    
    return {
        "aurc": aurc,
        "e_aurc": e_aurc
    }

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

def compute_sgr_coverage_at_risk(confidences, preds, targets, target_risks=[0.01, 0.05, 0.10],
                                 calib=None, delta=0.05):
    """
    Cobertura retida ao travar um risco máximo (ex: 1%, 5%, 10%) via SGR.

    ATENÇÃO — duas quantidades diferentes:

      * calib=None (padrão, comportamento histórico): o limiar é escolhido
        sobre os MESMOS arrays que são avaliados. Isso é um ponto de operação
        IN-SAMPLE ("cobertura atingível"), NÃO um certificado — a garantia de
        Geifman & El-Yaniv (2017) pressupõe um conjunto de calibração i.i.d.
        independente do conjunto avaliado. Serve para comparar métodos entre si
        sob procedimento idêntico, e é o que gerou as tabelas já publicadas.

      * calib=(conf_cal, preds_cal, targets_cal): o limiar é ajustado no split
        de calibração (validação ID) e aplicado sem alteração ao conjunto de
        teste. ESTA é a variante coberta pela garantia.

    delta continua 0.05 por padrão para não alterar silenciosamente números já
    reportados; passe delta=0.001 para a versão mais conservadora.
    """
    from methods.sgr.sgr import SGRController
    sgr = SGRController(delta=delta)

    results = {}
    suffix = "" if calib is None else "_heldout"
    for r_star in target_risks:
        if calib is None:
            theta, bound, coverage = sgr.fit(confidences, preds, targets, r_star)
        else:
            conf_c, preds_c, targ_c = calib
            theta, bound, _ = sgr.fit(conf_c, preds_c, targ_c, r_star)
            coverage = float((confidences >= theta).mean()) if theta != float('inf') else 0.0
        results[f"sgr_coverage_at_risk_{int(r_star*100)}{suffix}"] = coverage

    return results

def evaluate_selective(confidences, preds, targets):
    """Agregador das métricas de predição seletiva."""
    if isinstance(confidences, torch.Tensor): confidences = confidences.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor): preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()
    
    metrics = compute_aurc(confidences, preds, targets)
    metrics.update(compute_risk_at_coverage(confidences, preds, targets))
    metrics.update(compute_sgr_coverage_at_risk(confidences, preds, targets))
    return metrics
