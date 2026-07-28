"""
Benchmark real (não ilustrativo) de wall-clock, para responder à pergunta
"quanto tempo o kNN realmente economiza frente ao Sinkhorn?".

Compara duas variantes:
  1. sinkhorn_ot   -- reimplementação de referência do Sinkhorn-Knopp
                      (Cuturi, 2013), padrão, vetorizada em GPU. Não é o
                      código dos autores originais do LAQDA (não vendorizado
                      neste repositório); serve apenas de referência de
                      benchmark para a mesma classe de problema (plano de
                      transporte C x N).
  2. knn_forward   -- chamada real e completa de
                      TransductiveQDASampler.forward(), tal como
                      efetivamente usada em treino neste repositório.
                      Totalmente vetorizada (topk + divisão inteira para a
                      métrica de diagnóstico + gather via indexação
                      avançada), sem laços em Python.

Tamanhos de episódio: reais, extraídos de configs/methods_config.yaml e
configs/ood_splits.json (IntentPTCorpus: C=48 classes ID no fold 01, K=5,
q efetivo=10 após o teto max_query_total=500; RulingBRCorpus: C=10, K=5,
q efetivo=25, sem teto).
"""
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/user_beatrizalmeida/selective-risk-framework")
from methods.laqda.models.qda_sampler import TransductiveQDASampler

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
HIDDEN = 768
RHO = 15
T_SINKHORN = 100
N_WARMUP = 5
N_REPEATS = 30

torch.manual_seed(0)


def sinkhorn_ot(prototypes, queries, n_iters=T_SINKHORN, epsilon=0.1):
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
    return plan @ queries / (plan.sum(dim=1, keepdim=True) + 1e-8)


def bench(fn, *args, n_repeats=N_REPEATS, n_warmup=N_WARMUP):
    for _ in range(n_warmup):
        fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        fn(*args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_repeats * 1000


SCENARIOS = [
    ("IntentPTCorpus (C=48,K=5,q=10)", 48, 5, 10),
    ("RulingBRCorpus (C=10,K=5,q=25)", 10, 5, 25),
]

print(f"Dispositivo: {DEVICE}\n")
header = f"{'Cenário':34s} {'Sinkhorn (ms)':>14s} {'kNN forward() real (ms)':>24s}"
print(header)

results = {}
for name, C, K, Q in SCENARIOS:
    support = torch.randn(C * K, HIDDEN, device=DEVICE)
    query = torch.randn(C * Q, HIDDEN, device=DEVICE)
    prototypes = support.view(C, K, HIDDEN).mean(dim=1)
    sampler = TransductiveQDASampler(HIDDEN, C, K, Q, k=RHO).to(DEVICE)

    sinkhorn_ms = bench(sinkhorn_ot, prototypes, query)
    knn_full_ms = bench(lambda: sampler(support, query))

    results[name] = (sinkhorn_ms, knn_full_ms)
    print(f"{name:34s} {sinkhorn_ms:14.3f} {knn_full_ms:24.3f}")

print("\nRazão Sinkhorn / kNN forward() real (speedup do kNN):",
      {k: round(v[0] / v[1], 1) for k, v in results.items()})
