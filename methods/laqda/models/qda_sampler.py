import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_prototype_reestimation(prototypes, queries, rho: int, n_iters: int = 100, epsilon: float = 0.1):
    """Reestimação de protótipos via Transporte Ótimo entrópico (Sinkhorn-Knopp),
    fiel à formulação original do LAQDA (Liu et al., 2024): o lote inteiro de
    consultas é acoplado globalmente aos protótipos sob conservação de massa
    uniforme. O protótipo final mistura suporte e baricentro OT das consultas na
    mesma proporção de massa usada pelo caminho kNN (suporte 1/(1+rho),
    consultas rho/(1+rho)), para que as duas variantes fiquem comparáveis.
    Mesma parametrização de scripts/bench_sinkhorn_vs_knn.py.
    """
    C = prototypes.shape[0]
    N = queries.shape[0]
    p_norm = F.normalize(prototypes, p=2, dim=1)
    q_norm = F.normalize(queries, p=2, dim=1)
    cost = 1.0 - p_norm @ q_norm.T

    K = torch.exp(-cost / epsilon)
    u = torch.ones(C, device=prototypes.device) / C
    v = torch.ones(N, device=prototypes.device) / N
    a = torch.ones(C, device=prototypes.device) / C
    b = torch.ones(N, device=prototypes.device) / N

    for _ in range(n_iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)

    plan = torch.diag(u) @ K @ torch.diag(v)
    barycenter = plan @ queries / (plan.sum(dim=1, keepdim=True) + 1e-8)
    w_support = 1.0 / (1.0 + rho)
    return w_support * prototypes + (1.0 - w_support) * barycenter


class TransductiveQDASampler(nn.Module):
    """
    Sampler Transdutivo QDA utilizado no LAQDA.
    Estima a variância transdutivamente usando o query set.
    """
    def __init__(self, hidden_size: int, nway: int, kshot: int, qshot: int, k: int = 5):
        super(TransductiveQDASampler, self).__init__()
        self.dim = hidden_size
        self.nway = nway
        self.kshot = kshot
        self.qshot = qshot
        self.k = k

    def forward(self, support_emb, query_emb):
        # Cosine similarity for initial nearest neighbor matching
        s_norm = F.normalize(support_emb, p=2, dim=1)
        q_norm = F.normalize(query_emb, p=2, dim=1)
        similarity = torch.mm(s_norm, q_norm.transpose(0, 1))

        similarity = similarity.view(self.nway, self.kshot, -1)
        _, indices = similarity.topk(self.k, dim=2, largest=True, sorted=True)

        # Vectorized diagnostic accuracy: query_emb is laid out as `nway`
        # contiguous blocks of `qshot` queries each, so integer-dividing a
        # neighbor index by qshot recovers its class directly.
        neighbor_class = indices // self.qshot
        own_class = torch.arange(self.nway, device=indices.device).view(self.nway, 1, 1)
        acc_tensor = (neighbor_class == own_class).float().mean()

        nindices = indices.reshape(-1, self.k)
        convex_feat = query_emb[nindices]

        sampled_data = convex_feat.view(self.nway, self.kshot * self.k, self.dim)
        return sampled_data, acc_tensor
