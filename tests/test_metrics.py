import torch
import sys

from methods.baselines.maximum_softmax_prob.scorer import MSPScorer
from methods.baselines.energy_score.scorer import EnergyScorer
from methods.baselines.distance.mahalanobis.scorer import MahalanobisScorer
from methods.baselines.distance.knn.scorer import KNNScorer

def test_all():
    print("=== Testando Métricas OOD ===")
    
    N_train = 100
    N_test = 50
    D = 128
    C = 5

    # Dados Sintéticos
    train_features = torch.randn(N_train, D)
    train_labels = torch.randint(0, C, (N_train,))
    test_features = torch.randn(N_test, D)
    
    test_logits = torch.randn(N_test, C)

    # 1. MSP
    print("\n1. Testando Maximum Softmax Probability (MSP)...")
    msp = MSPScorer()
    scores = msp.compute_score(test_logits)
    print(f"MSP Output Shape: {scores.shape} (Esperado: [{N_test}])")
    assert scores.shape == (N_test,), "Falha no shape do MSP!"
    
    # 2. Energy
    print("2. Testando Energy Score...")
    energy = EnergyScorer(temperature=1.0)
    scores = energy.compute_score(test_logits)
    print(f"Energy Output Shape: {scores.shape} (Esperado: [{N_test}])")
    assert scores.shape == (N_test,), "Falha no shape do Energy!"
    
    # 3. Mahalanobis
    print("3. Testando Distância de Mahalanobis...")
    maha = MahalanobisScorer()
    maha.fit(train_features, train_labels)
    scores = maha.compute_score(test_features)
    print(f"Mahalanobis Output Shape: {scores.shape} (Esperado: [{N_test}])")
    assert scores.shape == (N_test,), "Falha no shape do Mahalanobis!"
    
    # 4. KNN
    print("4. Testando K-Nearest Neighbors (KNN)...")
    knn = KNNScorer(k=5)
    knn.fit(train_features)
    scores = knn.compute_score(test_features)
    print(f"KNN Output Shape: {scores.shape} (Esperado: [{N_test}])")
    assert scores.shape == (N_test,), "Falha no shape do KNN!"

    print("\n[SUCESSO] Todas as métricas foram instanciadas e testadas com tensores com sucesso!")

if __name__ == "__main__":
    try:
        test_all()
    except Exception as e:
        print(f"Erro durante o teste: {e}")
        sys.exit(1)
