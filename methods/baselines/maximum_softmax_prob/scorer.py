import torch
import torch.nn.functional as F

class MSPScorer:
    """
    Maximum Softmax Probability (MSP) [Hendrycks & Gimpel, 2017].
    Calcula a probabilidade máxima após aplicar Softmax nas logits.
    Valores mais baixos de MSP indicam maior incerteza (potencial OOD).
    """
    def __init__(self):
        pass

    def compute_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calcula o score MSP.
        Args:
            logits (torch.Tensor): Tensor de formato (N, num_classes)
        Returns:
            torch.Tensor: Tensor de formato (N,) com as probabilidades máximas.
        """
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        return max_probs
