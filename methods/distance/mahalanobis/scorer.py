import torch
import numpy as np

class MahalanobisScorer:
    """
    Mahalanobis Distance [Lee et al., 2018].
    Usa um espaço latente para ajustar uma distribuição Gaussiana por classe
    (com covariância empírica compartilhada) e usa a distância mínima aos centróides
    como métrica de confiança.
    Distâncias menores = maior confiança (ID).
    Nessa implementação, invertemos o sinal para que MAIORES valores = maior confiança.
    """
    def __init__(self):
        self.class_means = None
        self.precision_matrix = None

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor):
        """
        Ajusta os parâmetros (médias e precisão) com os dados de treinamento.
        Args:
            train_features: Tensor de formato (N, D)
            train_labels: Tensor de formato (N,) com IDs das classes (0 a C-1)
        """
        classes = torch.unique(train_labels)
        D = train_features.shape[1]
        
        # Calcular média de cada classe
        class_means = []
        for c in classes:
            c_features = train_features[train_labels == c]
            class_means.append(c_features.mean(dim=0))
        self.class_means = torch.stack(class_means)  # (C, D)
        
        # Calcular Matriz de Covariância compartilhada (Tied Covariance)
        cov = torch.zeros((D, D), device=train_features.device)
        for c, mean_vec in zip(classes, class_means):
            c_features = train_features[train_labels == c]
            centered = c_features - mean_vec
            # (N_c, D) -> (D, D)
            cov += torch.mm(centered.t(), centered)
        
        cov = cov / train_features.shape[0]
        
        # Adicionar jitter para estabilidade numérica
        epsilon = 1e-6
        cov += torch.eye(D, device=cov.device) * epsilon
        
        # Inverter para obter a Matriz de Precisão
        self.precision_matrix = torch.linalg.inv(cov)

    def compute_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcula a distância Mahalanobis invertida (-dist) para usar como score de confiança.
        Args:
            features: Tensor de formato (N, D)
        Returns:
            scores: Tensor de formato (N,) com as confianças.
        """
        if self.class_means is None or self.precision_matrix is None:
            raise ValueError("O MahalanobisScorer deve ser ajustado com fit() antes de gerar scores.")
            
        N = features.shape[0]
        C = self.class_means.shape[0]
        
        scores = []
        for i in range(N):
            f = features[i]
            dists = []
            for c in range(C):
                diff = f - self.class_means[c]
                # Mahalanobis dist = diff^T * P * diff
                left = torch.matmul(diff, self.precision_matrix)
                dist = torch.matmul(left, diff)
                dists.append(dist)
            
            # A classe mais próxima define a distância final
            min_dist = torch.min(torch.stack(dists))
            # Inverter o sinal para que distâncias menores = scores MAIORES (mais confidentes)
            scores.append(-min_dist)
            
        return torch.stack(scores)
