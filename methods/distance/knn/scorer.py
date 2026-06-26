import torch

class KNNScorer:
    """
    K-Nearest Neighbors (KNN) OOD Detection [Sun et al., 2022].
    Memoriza os embeddings de treino e avalia a distância Euclidiana
    ou Cosseno média para os K vizinhos mais próximos.
    Valores MAIORES indicam maior confiança.
    Nessa implementação, calculamos a distância Euclidiana média e invertemos o sinal.
    Se o espaço for contrastivo (como no LAQDA), KNN Contrastive ganha eficiência
    espacial naturalmente.
    """
    def __init__(self, k: int = 50):
        self.k = k
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
        Calcula a distância média para os K vizinhos mais próximos e inverte o sinal.
        Args:
            features: Tensor de formato (M, D)
        Returns:
            scores: Tensor de formato (M,) com as confianças (-dist).
        """
        if self.bank is None:
            raise ValueError("O KNNScorer deve ser ajustado com fit() antes de gerar scores.")
            
        # Calcular distâncias quadradas par-a-par: ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
        feat_norm = (features ** 2).sum(dim=1, keepdim=True)
        bank_norm = (self.bank ** 2).sum(dim=1, keepdim=True).t()
        
        dists = feat_norm + bank_norm - 2 * torch.mm(features, self.bank.t())
        # Evitar valores negativos por precisão flutuante
        dists = torch.clamp(dists, min=0.0)
        dists = torch.sqrt(dists)
        
        # Encontrar as K menores distâncias para cada feature (dim=1)
        topk_dists, _ = torch.topk(dists, k=self.k, dim=1, largest=False)
        
        # A distância do KNN é a média das distâncias aos K vizinhos
        knn_dist = topk_dists.mean(dim=1)
        
        # Inverter para que distâncias menores sejam traduzidas para confiança maior
        scores = -knn_dist
        
        return scores
