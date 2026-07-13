import torch
import numpy as np
from sklearn.covariance import LedoitWolf

class MahalanobisScorer:
    """
    Scorer Baseado em Distância de Mahalanobis - "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (Lee et al., NeurIPS 2018).
    
    INTUIÇÃO DO PAPER:
    O método propõe que o espaço latente de redes neurais profundas bem treinadas pode ser bem aproximado
    por uma Mistura de Gaussianas (Gaussian Mixture Model) onde cada classe tem sua própria média (centróide),
    mas todas compartilham a mesma matriz de covariância. 
    
    A Distância de Mahalanobis, ao contrário da distância Euclidiana simples, leva em consideração essa
    matriz de covariância (ou seja, a correlação e a escala entre as diferentes features latentes).
    Assim, uma amostra teste é classificada computando sua Distância de Mahalanobis até o centróide mais próximo.
    Amostras OOD estarão muito mais distantes da distribuição multivariada ID, resultando em grandes distâncias.
    
    Nesta implementação, calculamos a distância mínima e a invertemos (multiplicamos por -1) para que, 
    assim como nos outros scorers, valores MAIORES representem maior confiança ID (In-Distribution).
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
        # Usamos o Shrinkage Estimator de Ledoit-Wolf para evitar 
        # matrizes singulares no regime few-shot (K <= 10).
        centered_features = []
        for c, mean_vec in zip(classes, class_means):
            c_features = train_features[train_labels == c]
            centered = c_features - mean_vec
            centered_features.append(centered)
            
        centered_features = torch.cat(centered_features).cpu().numpy()
        
        lw = LedoitWolf()
        lw.fit(centered_features)
        
        # Inverter para obter a Matriz de Precisão (LedoitWolf já fornece precision_)
        self.precision_matrix = torch.tensor(lw.precision_, dtype=torch.float32, device=train_features.device)

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
