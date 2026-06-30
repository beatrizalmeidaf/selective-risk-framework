import torch
import torch.nn.functional as F

class KNNScorer:
    """
    Scorer Baseado em K-Nearest Neighbors (KNN) - "KNN-Contrastive Learning for Out-of-Domain Intent Classification".
    
    INTUIÇÃO DO PAPER:
    Diferente dos métodos paramétricos que modelam a distribuição usando estatísticas fixas (como a matriz de covariância
    no Mahalanobis), o KNN assume uma abordagem não-paramétrica. No contexto de Aprendizado Contrastivo para PNL 
    (Intent Classification), o espaço latente é otimizado para agrupar intenções similares e afastar intenções diferentes.
    
    Amostras Out-of-Domain (OOD) tendem a cair em regiões esparsas do espaço de features, muito distantes dos agrupamentos
    de treinamento In-Domain (ID). Ao memorizar todo o conjunto de treinamento, podemos medir a distância (Euclidiana 
    ou Similaridade do Cosseno) média aos K vizinhos mais próximos. 
    
    Se o dado for ID, seus vizinhos estarão extremamente próximos devido ao viés contrastivo. Se for OOD, a distância 
    média para os K vizinhos será grande. Nessa implementação, multiplicamos as distâncias por -1 (ou usamos a similaridade 
    pura) para que valores MAIORES sempre indiquem maior confiança (certeza de ser ID).
    """
    def __init__(self, k: int = 50, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        self.bank = None

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor = None):
        """
        Salva o banco de features. Labels são opcionais para KNN não-supervisionado OOD.
        Args:
            train_features: Tensor de formato (N, D)
        """
        # Normalizar features pode ajudar no KNN, mas manteremos o espaço original
        # para preservar o design de contraste Euclidiano do LAQDA.
        self.bank = train_features.clone().detach()

    def compute_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcula a distância ou similaridade para os K vizinhos mais próximos.
        Args:
            features: Tensor de formato (M, D)
        Returns:
            scores: Tensor de formato (M,) com as confianças.
        """
        if self.bank is None:
            raise ValueError("O KNNScorer deve ser ajustado com fit() antes de gerar scores.")
            
        if self.metric == 'cosine':
            # Normalização L2 para usar similaridade angular
            feat_norm = F.normalize(features, p=2, dim=1)
            bank_norm = F.normalize(self.bank, p=2, dim=1)
            
            # Similaridade do cosseno: feat_norm * bank_norm.T -> (M, N)
            sims = torch.mm(feat_norm, bank_norm.t())
            
            # K maiores similaridades (largest=True)
            k_safe = min(self.k, sims.size(1))
            topk_sims, _ = torch.topk(sims, k=k_safe, dim=1, largest=True)
            
            # Confiança é a similaridade média (já é maior para ID, então mantemos positivo)
            scores = topk_sims.mean(dim=1)
            return scores
            
        elif self.metric == 'euclidean':
            # Calcular distâncias quadradas par-a-par
            feat_norm = (features ** 2).sum(dim=1, keepdim=True)
            bank_norm = (self.bank ** 2).sum(dim=1, keepdim=True).t()
            
            dists = feat_norm + bank_norm - 2 * torch.mm(features, self.bank.t())
            dists = torch.clamp(dists, min=0.0)
            dists = torch.sqrt(dists)
            
            # K menores distâncias
            k_safe = min(self.k, dists.size(1))
            topk_dists, _ = torch.topk(dists, k=k_safe, dim=1, largest=False)
            
            knn_dist = topk_dists.mean(dim=1)
            scores = -knn_dist
            
            return scores
        else:
            raise ValueError(f"Métrica não suportada: {self.metric}")
