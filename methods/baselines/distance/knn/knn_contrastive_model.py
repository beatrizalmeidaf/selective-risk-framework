import torch
import torch.nn as nn
from methods.baselines.models.standard_classifier import BaselineClassifier

class KNNContrastiveModel(nn.Module):
    """
    Implementação baseada em MoCo (Momentum Contrast) adaptada para Few-Shot.
    Possui dois encoders: Query Encoder e Key (Momentum) Encoder.
    """
    def __init__(self, model_name: str, num_classes: int, dim: int = 768, queue_size: int = 512, momentum: float = 0.999):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum

        # Encoder principal (treinado com gradiente)
        self.encoder_q = BaselineClassifier(model_name, num_classes)
        
        # Encoder de momento (treinado via EMA)
        self.encoder_k = BaselineClassifier(model_name, num_classes)
        
        # Inicializa ambos com os mesmos pesos e desativa gradientes no encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Criar a queue de features (dim, queue_size)
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        # Criar a queue de labels correspondentes (queue_size,)
        self.register_buffer("queue_labels", -torch.ones(queue_size, dtype=torch.long))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gathered keys before updating queue
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f"queue_size {self.queue_size} must be divisible by batch_size {batch_size}"

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_labels[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, texts, labels=None):
        """
        Input:
            texts: lista de textos (batch)
            labels: rótulos das classes (apenas no treino para a queue)
        Output:
            features_q, logits_q (para cross-entropy e avaliação)
            E, durante o treino: keys e queue elements para a KNN-Contrastive Loss.
        """
        features_q, logits_q = self.encoder_q(texts)
        features_q = nn.functional.normalize(features_q, dim=1)

        if not self.training or labels is None:
            return features_q, logits_q

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            features_k, _ = self.encoder_k(texts)
            features_k = nn.functional.normalize(features_k, dim=1)

        # Retorna os itens do batch atual (queries e chaves) e a queue
        return features_q, logits_q, features_k, self.queue.clone().detach(), self.queue_labels.clone().detach()
