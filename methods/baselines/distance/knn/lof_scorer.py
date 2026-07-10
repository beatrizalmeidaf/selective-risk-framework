import torch
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize

class LOFScorer:
    """
    Local Outlier Factor (LOF) Scorer.
    Usado como detector OOD no KNN-Contrastive Learning (Zhou et al.).
    
    O LOF mede a desviação local de densidade de uma amostra em relação a seus vizinhos.
    O score_samples() do scikit-learn retorna o LOF negativo (valores maiores indicam inliers / ID).
    """
    def __init__(self, n_neighbors: int = 20, metric: str = 'cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        # novelty=True permite usar o LOF para predição em novos dados após o fit
        self.clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric, novelty=True)
        self.is_fitted = False

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor = None):
        """
        Treina o LOF usando features do conjunto de treino.
        """
        features_np = train_features.cpu().numpy()
        
        if self.metric == 'cosine':
            # Prevenir erros numéricos normalizando as features antes
            features_np = normalize(features_np, norm='l2', axis=1)
            
        self.clf.fit(features_np)
        self.is_fitted = True

    def compute_score(self, test_features: torch.Tensor) -> torch.Tensor:
        """
        Calcula o score de confiança LOF para cada amostra.
        Score maior = Maior densidade local = Mais confiante (In-Domain).
        """
        if not self.is_fitted:
            raise RuntimeError("O LOFScorer precisa ser ajustado com .fit() antes do .compute_score()")
            
        features_np = test_features.cpu().numpy()
        
        if self.metric == 'cosine':
            features_np = normalize(features_np, norm='l2', axis=1)
            
        # O score_samples retorna o negative outlier factor
        scores = self.clf.score_samples(features_np)
        return torch.tensor(scores, dtype=torch.float32)
