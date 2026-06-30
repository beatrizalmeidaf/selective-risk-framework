import torch
import torch.nn as nn
import torch.nn.functional as F

class ConjNormScorer:
    """
    Scorer Baseado em Normalização Conjugada (ConjNorm).
    Baseado em "CONJNORM: TRACTABLE DENSITY ESTIMATION FOR OUT-OF-DISTRIBUTION DETECTION" (ICLR 2024).
    
    INTUIÇÃO DO PAPER:
    O artigo do ICLR 2024 propõe o ConjNorm, uma técnica tratável de estimativa de densidade para 
    detecção OOD. Ele demonstra que métodos baseados puramente em logit (como o Softmax ou Energy clássico) 
    falham ao tentar estimar a densidade do espaço latente pois ignoram a magnitude e a estrutura direcional
    das features. 
    
    A solução proposta pelo ConjNorm é normalizar as features e os pesos do classificador de forma a
    mapear os dados para uma hiperesfera (espaço de similaridade de cosseno). Ao fazer essa normalização
    conjugada, o modelo consegue realizar uma estimativa de densidade muito mais precisa, onde a distância 
    angular para o centróide da classe torna-se um proxy robusto para a probabilidade In-Distribution, 
    eliminando problemas de hiper-confiança causados por anomalias de magnitude.
    """
    def __init__(self, classifier: nn.Linear):
        self.classifier = classifier

    def compute_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcula o Maximum Cosine Similarity.
        Args:
            features (torch.Tensor): Tensores de tamanho (N, D)
        Returns:
            torch.Tensor: Confiança (N,). Maior similaridade = maior certeza (ID).
        """
        device = features.device
        # Extrai os pesos W originais treinados da camada linear (K, D)
        weights = self.classifier.weight.to(device)
        
        # Normalização L2: Força todos os vetores a habitarem a superfície de uma hiperesfera (norma = 1)
        f_norm = F.normalize(features, p=2, dim=-1)
        w_norm = F.normalize(weights, p=2, dim=-1)
        
        # Similaridade do Cosseno: Em vez do produto escalar irrestrito, calcula o alinhamento direcional.
        # Resultado está perfeitamente contido entre [-1, 1].
        # f_norm: (N, D) multiplicada por w_norm.T: (D, K) gera as novas logits de tamanho (N, K)
        cosine_sim = torch.mm(f_norm, w_norm.t())
        
        # NOTA SOBRE O BIAS: O viés (bias) original da camada linear é descartado propositalmente, 
        # pois ele foi otimizado para o espaço Euclidiano de magnitudes variáveis e prejudica a pureza do espaço angular.
        
        # A confiança OOD é simplesmente a similaridade máxima (o quão angularmente próximo o dado está da sua classe predita).
        max_cosine, _ = torch.max(cosine_sim, dim=-1)
        
        return max_cosine
