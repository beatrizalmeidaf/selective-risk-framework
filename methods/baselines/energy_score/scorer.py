import torch

class EnergyScorer:
    """
    Energy Score [Liu et al., 2020].
    Usa a função de energia de modelos baseados em EBM (Energy-Based Models).
    Valores MAIORES indicam maior certeza (ID), e valores menores indicam OOD.
    Pode-se mudar o sinal dependendo da formulação de detecção.
    A formulação tradicional avalia `-T * logsumexp(logits / T)`.
    Nessa implementação, retornamos `T * logsumexp(logits / T)` de forma que
    valores maiores são mais "confidentes" (para parear com o MSP).
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
