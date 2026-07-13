"""
Fonte única de verdade para o score de confiança do LAQDA.

A calibração do SGR (validação, evaluator.py) e a aplicação do threshold
(teste, inference/infer.py) DEVEM usar exatamente a mesma função de score.
Qualquer divergência de métrica de distância, normalização ou escala (tau)
invalida silenciosamente o threshold transferido: um limiar calibrado em
margem euclidiana (~10-160) aplicado sobre cosseno x tau (limitado a [-15, 15])
rejeita 100% do conjunto de teste.
"""
import torch
import torch.nn.functional as F

TAU = 15.0


def compute_prototype_scores(query_embeddings: torch.Tensor,
                             prototypes: torch.Tensor,
                             tau: float = TAU):
    """
    Score canônico: similaridade de cosseno (embeddings L2-normalizados) x tau.

    Args:
        query_embeddings: [N, D] embeddings das consultas.
        prototypes:       [C, D] protótipos de classe (indutivos: média do suporte).
        tau:              temperatura de escala (padrão 15.0).

    Returns:
        dists:        [N, C] distâncias (-sim * tau); menor = mais próximo.
        preds:        [N]    índice do protótipo mais próximo.
        conf_default: [N]    -min_dist = max sim * tau (score DEFAULT; escala do SGR).
        conf_margin:  [N]    2ª menor dist - menor dist (score MARGIN, diagnóstico;
                             NÃO compatível com thresholds calibrados no DEFAULT).
    """
    q = F.normalize(query_embeddings, p=2, dim=1)
    p = F.normalize(prototypes, p=2, dim=1)
    sim = torch.mm(q, p.t())
    dists = -sim * tau

    sorted_dists, sorted_idx = torch.sort(dists, dim=1)
    preds = sorted_idx[:, 0]
    conf_default = -sorted_dists[:, 0]

    if dists.shape[1] >= 2:
        conf_margin = sorted_dists[:, 1] - sorted_dists[:, 0]
    else:
        conf_margin = conf_default

    return dists, preds, conf_default, conf_margin
