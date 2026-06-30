import torch

class GradNormScorer:
    """
    Scorer Baseado em GradNorm - "On the Importance of Gradients for Out-of-Distribution Detection" (Huang et al., NeurIPS 2021).
    
    INTUIÇÃO DO PAPER:
    O artigo mostra que o espaço de gradientes captura informações cruciais sobre a incerteza do modelo
    que o espaço das ativações (features) ou saídas normais (logits/softmax) acaba perdendo. Amostras ID (conhecidas)
    geram gradientes de perda maiores e mais concentrados quando comparadas a uma distribuição de ruído uniforme,
    enquanto amostras OOD produzem gradientes esparsos e de menor magnitude na última camada.
    
    A métrica proposta é calcular a norma L1 do gradiente da Divergência KL (KL-divergence) entre a saída da rede 
    e a distribuição uniforme (total incerteza) com respeito aos pesos da última camada (classificador linear).
    
    Para uma camada puramente linear (sem viés), a norma L1 desse gradiente pode ser fatorada de forma analítica exata,
    evitando a necessidade de rodar a custosa função `loss.backward()`:
    GradNorm = || softmax(logits) - 1/K ||_1 * || features ||_1
    """
    def __init__(self, num_classes: int):
        # Armazena o número total de classes K, necessário para a distribuição uniforme (1/K).
        self.num_classes = num_classes

    def compute_score(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Calcula o GradNorm score (post-hoc analítico).
        Args:
            features (torch.Tensor): Features extraídas da penúltima camada (N, D).
            logits (torch.Tensor): Logits produzidas pelo classificador (N, K).
        Returns:
            torch.Tensor: Scores de confiança (N,). (Maior = ID, Menor = OOD).
        """
        # Calcula as probabilidades com Softmax normal
        probs = torch.softmax(logits, dim=-1)
        
        # O termo |p - 1/K| mede a diferença entre as predições do modelo e a distribuição de incerteza máxima.
        # Norma L1 dessa diferença ao longo da dimensão das classes (dim=-1)
        diff_norm = torch.norm(probs - (1.0 / self.num_classes), p=1, dim=-1)
        
        # Norma L1 das ativações (features latentes extraídas antes da camada linear)
        feat_norm = torch.norm(features, p=1, dim=-1)
        
        # Pela fatoração de aproximação em tensores de dimensão cruzada, o gradiente analítico L1
        # da camada linear torna-se a simples multiplicação dessas duas normas.
        gradnorm = diff_norm * feat_norm
        
        return gradnorm
