import torch
import torch.nn as nn
import torch.nn.functional as F

class ConjNormScorer:
    """
    Scorer Baseado em Normalização Conjugada (ConjNorm).
    Baseado em "CONJNORM: TRACTABLE DENSITY ESTIMATION FOR OUT-OF-DISTRIBUTION DETECTION" (ICLR 2024).
    
    Reformulado para utilizar a Teoria de Divergência de Bregman com pares conjugados l_p e l_q, 
    onde a constante de partição (Partition Function) é estimada de forma tratável utilizando 
    Amostragem de Monte Carlo (Importance Sampling) sobre os dados de treino In-Domain.
    """
    def __init__(self, p: float = 2.5, alpha: float = 1.0):
        """
        Args:
            p (float): O coeficiente da norma lp (recomendado no artigo entre 2 e 3).
            alpha (float): Sampling ratio para o Importance Sampling da Partition Function. 
                           Para few-shot, recomenda-se 1.0 (usar todas as amostras disponíveis).
        """
        self.p = p
        self.q = p / (p - 1.0)
        self.alpha = alpha
        
        self.mu_k = None
        self.phi_k_log = None
        self.num_classes = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Ajusta os centróides e computa a constante de partição Phi(k) no espaço logarítmico.
        Args:
            features (torch.Tensor): Features de treino (N, D).
            labels (torch.Tensor): Rótulos de treino (N,).
        """
        device = features.device
        self.num_classes = int(labels.max().item()) + 1
        
        self.mu_k = torch.zeros(self.num_classes, features.size(1), device=device)
        
        # 1. Calcular os centróides mu_k para cada classe
        for k in range(self.num_classes):
            mask = (labels == k)
            if mask.sum() > 0:
                self.mu_k[k] = features[mask].mean(dim=0)
                
        # 2. Amostragem de Monte Carlo (Importance Sampling) para \Phi(k)
        num_samples = int(features.size(0) * self.alpha)
        num_samples = max(1, min(num_samples, features.size(0)))
        
        indices = torch.randperm(features.size(0), device=device)[:num_samples]
        sampled_features = features[indices]
        
        # 3. Calcular a Função de Partição Phi(k) via Média Empírica (espaço de log)
        self.phi_k_log = torch.zeros(self.num_classes, device=device)
        for k in range(self.num_classes):
            log_g_theta = self._compute_log_g_theta(sampled_features, k)
            
            # \Phi_{IS}(k) = \frac{1}{n} \sum g_\theta(z, k)
            # log( \Phi_{IS}(k) ) = logsumexp( log_g_theta ) - log(n)
            log_phi_k = torch.logsumexp(log_g_theta, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32, device=device))
            self.phi_k_log[k] = log_phi_k

    def _compute_log_g_theta(self, z: torch.Tensor, k: int) -> torch.Tensor:
        """
        Retorna log( g_\theta(z, k) ) = -d_\phi(z, \mu_k).
        d_\phi(z, \mu_k) = 0.5 * ||z||_q^q + 0.5 * ||\mu_k||_q^q - <z, sign(\mu_k) * |\mu_k|^(q-1)>
        """
        mu = self.mu_k[k]  # (D,)
        
        # ||z||_q^q
        z_q_norm_q = torch.sum(torch.abs(z) ** self.q, dim=-1)  # (N,)
        
        # ||\mu_k||_q^q
        mu_q_norm_q = torch.sum(torch.abs(mu) ** self.q, dim=-1)  # escalar
        
        # \nabla (0.5 * ||\mu_k||_q^q) = sign(\mu_k) * |\mu_k|^(q-1)
        grad_mu = torch.sign(mu) * (torch.abs(mu) ** (self.q - 1.0))  # (D,)
        
        # <z, \nabla>
        dot_product = torch.matmul(z, grad_mu)  # (N,)
        
        d_phi = 0.5 * z_q_norm_q + 0.5 * mu_q_norm_q - dot_product
        
        return -d_phi

    def compute_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcula o score de densidade ID para OOD Detection.
        S(z) = \log ( \sum_{k=1}^K \frac{g_\theta(z, k)}{\Phi(k)} )
        
        Args:
            features (torch.Tensor): Features de teste (N, D).
        Returns:
            torch.Tensor: Scores de confiança (N,). Valores maiores indicam In-Domain.
        """
        if self.mu_k is None or self.phi_k_log is None:
            raise RuntimeError("ConjNormScorer precisa ser treinado usando .fit() primeiro.")
            
        N = features.size(0)
        K = self.num_classes
        
        log_densities = torch.zeros(N, K, device=features.device)
        
        for k in range(K):
            log_g = self._compute_log_g_theta(features, k)  # (N,)
            # \log( g / \Phi ) = \log(g) - \log(\Phi)
            log_densities[:, k] = log_g - self.phi_k_log[k]
            
        # Score Agregado: S(z) = \log( \sum \exp(log_densities) )
        scores = torch.logsumexp(log_densities, dim=-1)
        
        return scores
