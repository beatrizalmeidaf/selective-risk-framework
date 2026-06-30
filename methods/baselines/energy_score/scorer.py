import torch

class EnergyScorer:
    """
    Scorer Baseado em Energy Score - "Energy-based Out-of-distribution Detection" 
    (Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li).
    
    INTUIÇÃO DO PAPER:
    Diferente do Softmax tradicional (MSP) que força a soma das probabilidades a ser 1, diluindo a diferença
    entre previsões confiáveis e uniformes, os Modelos Baseados em Energia (EBMs) mapeiam cada ponto de dado para 
    um escalar não-probabilístico chamado "Energia". 
    
    O paper prova teoricamente que a função de energia de uma rede neural (calculada através do logsumexp das logits)
    é naturalmente menor para dados de treinamento observados (In-Distribution) e maior para dados anômalos (OOD).
    Portanto, a "Energia" atua como uma pontuação universal de densidade sem as restrições matemáticas do Softmax.
    
    A formulação tradicional de Energia avalia `-T * logsumexp(logits / T)`.
    Nesta implementação, nós omitimos o sinal negativo e retornamos apenas `T * logsumexp(logits / T)`,
    para que valores MAIORES indiquem amostras ID (mantendo a paridade de direção com o MSP).
    """
    def __init__(self, temperature: float = 1.0):
        self.T = temperature

    def compute_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calcula o Energy Score.
        Args:
            logits (torch.Tensor): Tensor de formato (N, num_classes)
        Returns:
            torch.Tensor: Tensor de formato (N,) com a energia computada.
        """
        energy = self.T * torch.logsumexp(logits / self.T, dim=-1)
        return energy
