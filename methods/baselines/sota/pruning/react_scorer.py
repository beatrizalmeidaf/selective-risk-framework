import torch
import torch.nn as nn
import numpy as np

class ReActScorer:
    """
    Scorer Baseado em ReAct - "Rectified Activations for Out-of-Distribution Detection" (Sun et al., NeurIPS 2021).
    
    INTUIÇÃO DO PAPER:
    O artigo descobre que a principal razão para as redes neurais apresentarem alta confiança (hiper-confiança)
    em amostras OOD é a presença de unidades de ativação anômalas, com valores desproporcionalmente grandes (outliers)
    na penúltima camada da rede. Para amostras ID (In-Distribution), as ativações costumam ser bem comportadas, mas 
    dados muito anômalos "acendem" certas features com magnitudes absurdas, distorcendo o Softmax.
    
    A solução (ReAct) é puramente baseada no truncamento lógico (clipping) das ativações num limiar pré-definido `c`.
    O limiar `c` geralmente é escolhido como um percentil alto (ex: 90%) das ativações observadas nos dados de treinamento ID.
    Qualquer ativação que ultrapasse esse limite é podada (forçada para `c`), neutralizando a hiper-confiança OOD.
    Após a poda, a camada linear e o cálculo da Incerteza (via Energy Score) são executados normalmente.
    """
    def __init__(self, classifier: nn.Module, percentile: float = 90.0, temperature: float = 1.0):
        self.classifier = classifier
        self.percentile = percentile
        self.temperature = temperature
        self.c = None

    def fit(self, train_features: torch.Tensor):
        """
        Aprende o limiar `c` com base na distribuição das ativações do conjunto In-Distribution (ID).
        O valor `c` age como um teto máximo de magnitude permitida para as representações latentes.
        """
        # Achatamos todas as dimensões do tensor de treino para extrair o percentil global da distribuição latente
        features_np = train_features.cpu().numpy().flatten()
        self.c = np.percentile(features_np, self.percentile)
        
    def compute_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aplica o truncamento lógico (clipping) nas features e retorna a certeza recalculada pelo Energy Score.
        """
        if self.c is None:
            raise ValueError("ReActScorer precisa ser ajustado com fit() (usando dados de treino) antes da inferência.")
            
        device = features.device
        classifier_device = next(self.classifier.parameters()).device
        
        # O Clipping fundamental do ReAct: trunca o teto máximo das ativações no valor de `c`.
        # Isso não afeta o funcionamento regular da rede em dados ID (pois a maioria fica abaixo de c),
        # mas destrói completamente o viés das features gigantes do OOD.
        clipped_features = torch.clamp(features, max=float(self.c))
        
        with torch.no_grad():
            # Passamos as features retificadas pela camada linear existente para obter novas logits sem a distorção.
            new_logits = self.classifier(clipped_features.to(classifier_device))
            
        # Para OOD Scorer, o ReAct tradicionalmente utiliza o Energy Score sobre essas logits corrigidas.
        energy = self.temperature * torch.logsumexp(new_logits / self.temperature, dim=-1)
        
        return energy.to(device)
