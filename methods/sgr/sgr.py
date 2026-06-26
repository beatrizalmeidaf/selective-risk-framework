import numpy as np
import torch
import math
from scipy.stats import beta

class SGRController:
    """
    Selection with Guaranteed Risk (SGR)
    Baseado em Geifman & El-Yaniv (2017) - "Selective Classification for Deep Neural Networks"
    
    Implementa o Algoritmo 1: Busca binária para achar o limiar ótimo (theta) 
    que garante um risco máximo aceitável (r_star) com confiança (1 - delta).
    """
    def __init__(self, delta=0.001):
        self.delta = delta

    def _gascuel_bound(self, r_hat, m, delta):
        """
        Calcula B*(r_hat, delta, m) usando a inversão da cauda Binomial (Clopper-Pearson).
        Conforme o Lema 3.1 do paper (Gascuel and Caraux, 1992).
        """
        k = int(math.floor(m * r_hat))
        if m == 0:
            return 1.0 # Risco máximo (incerteza total)
            
        # Pela distribuição Beta (intervalo exato de Clopper-Pearson):
        # b = Beta.ppf(1 - delta, k + 1, m - k)
        b_star = beta.ppf(1.0 - delta, k + 1, m - k)
        
        # Correção para o caso onde todos estão corretos (k=0, falha de precisão na Beta em algumas bibliotecas)
        if math.isnan(b_star):
            if k == 0:
                b_star = 1.0 - math.pow(delta, 1.0 / m)
            else:
                b_star = 1.0
                
        return b_star

    def fit(self, confidences, preds, targets, r_star):
        """
        Algoritmo 1: SGR(f, k_f, delta, r_star, S_m)
        Retorna o threshold (theta_star) ótimo que garante o risco r_star.
        """
        if isinstance(confidences, torch.Tensor): confidences = confidences.cpu().numpy()
        if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor): targets = targets.cpu().numpy()

        m_total = len(confidences)
        
        # 1. Ordenar S_m de acordo com k_f(x_i) (ordem DECRESCENTE de confiança)
        # Atenção: O paper sugere ordenar e usar o índice z, aqui faremos a busca binária
        # nos próprios valores ordenados únicos de confiança para ser robusto a empates.
        unique_confs = np.unique(confidences)
        unique_confs = np.sort(unique_confs) # Crescente
        
        # Se m_total for pequeno, a busca binária é sobre os índices
        k_iters = math.ceil(math.log2(m_total)) if m_total > 1 else 1
        
        z_min = 0
        z_max = len(unique_confs) - 1
        
        best_theta = float('inf')
        best_bound = 1.0
        best_coverage = 0.0
        
        for i in range(k_iters):
            z = math.ceil((z_min + z_max) / 2)
            if z > len(unique_confs) - 1:
                break
                
            theta = unique_confs[z]
            
            # g_i(S_m): projetar apenas amostras com confiança >= theta
            accepted_mask = confidences >= theta
            m_i = np.sum(accepted_mask)
            
            if m_i == 0:
                # Rejeitou tudo
                r_hat_i = 0.0
                b_star_i = 1.0
            else:
                accepted_preds = preds[accepted_mask]
                accepted_targets = targets[accepted_mask]
                errors = np.sum(accepted_preds != accepted_targets)
                r_hat_i = errors / m_i
                
                # Lemma 3.1: b_star_i = B*(r_hat_i, delta/k, m_i)
                b_star_i = self._gascuel_bound(r_hat_i, m_i, self.delta / k_iters)
                
            if b_star_i < r_star:
                # O limite de risco é menor que o alvo! Podemos tentar diminuir theta (aumentar z_max em índice invertido? Não)
                # Queremos maximizar a cobertura, ou seja, MENOR theta possível que ainda atenda a condição.
                # Se atendeu, o theta atual é seguro, vamos tentar um theta MENOR (índices menores)
                best_theta = theta
                best_bound = b_star_i
                best_coverage = m_i / m_total
                z_max = z - 1
            else:
                # O limite estourou. Precisamos de confianças MAIORES (theta maior)
                z_min = z + 1
                
            if z_min > z_max:
                break
                
        return best_theta, best_bound, best_coverage
