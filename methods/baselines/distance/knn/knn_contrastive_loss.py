import torch
import torch.nn as nn
import torch.nn.functional as F

class KNNContrastiveLoss(nn.Module):
    """
    KNN-Contrastive Loss (Zhou et al.)
    Seleciona os top-K vizinhos mais próximos da MESMA CLASSE como amostras positivas.
    Amostras de outras classes atuam como negativas.
    """
    def __init__(self, k_neighbors: int = 5, tau: float = 0.07):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.tau = tau

    def forward(self, features_q, features_k, queue, queue_labels, labels):
        """
        features_q: (batch_size, dim) - L2 normalized
        features_k: (batch_size, dim) - L2 normalized
        queue: (dim, queue_size) - L2 normalized
        queue_labels: (queue_size,)
        labels: (batch_size,) - rótulos reais das features_q
        """
        batch_size = features_q.size(0)
        
        # Concatena keys do batch atual e da queue para formar a base de busca
        # all_keys: (batch_size + queue_size, dim)
        all_keys = torch.cat([features_k, queue.T], dim=0)
        
        # all_labels: (batch_size + queue_size,)
        all_labels = torch.cat([labels, queue_labels], dim=0)
        
        # similarity matrix (batch_size, batch_size + queue_size)
        sim_matrix = torch.matmul(features_q, all_keys.T) / self.tau
        
        loss = 0.0
        
        for i in range(batch_size):
            q_label = labels[i]
            
            # Máscara de mesma classe (exclui o próprio elemento i do batch para evitar self-matching trivial,
            # embora features_q e features_k sejam de encoders diferentes, podemos manter o key correspondente)
            is_same_class = (all_labels == q_label)
            is_diff_class = (all_labels != q_label) & (all_labels != -1)  # Ignora -1 (espaço vazio da fila)
            
            # Remover o próprio índice (features_k correspondente na posição i)
            is_same_class[i] = False
            
            valid_pos_indices = torch.where(is_same_class)[0]
            valid_neg_indices = torch.where(is_diff_class)[0]
            
            # Se não houver amostras positivas válidas suficientes, ignoramos (raro após primeiras batches)
            if len(valid_pos_indices) == 0:
                continue
                
            # Seleciona as k maiores similaridades da mesma classe
            pos_sims = sim_matrix[i, valid_pos_indices]
            
            k_actual = min(self.k_neighbors, len(pos_sims))
            topk_pos_sims, _ = torch.topk(pos_sims, k_actual)
            
            # Seleciona as similaridades das outras classes (negativas)
            neg_sims = sim_matrix[i, valid_neg_indices]
            
            # Para cada positivo dos Top-K, calcula a InfoNCE
            # Loss_i = \sum_{j \in TopK} -log( exp(sim_pos) / (exp(sim_pos) + \sum exp(sim_negs)) )
            
            # Para otimizar, o denominador é compartilhado
            sum_neg_exp = torch.sum(torch.exp(neg_sims))
            
            # Proteção contra NaN (denominador pode ficar grande, log-sum-exp é mais estável numéricamente)
            # -log( e^p / (e^p + N) ) = -p + log(e^p + N)
            term_loss = -topk_pos_sims + torch.log(torch.exp(topk_pos_sims) + sum_neg_exp + 1e-8)
            
            loss += term_loss.mean()
            
        return loss / batch_size if batch_size > 0 else torch.tensor(0.0, requires_grad=True).to(features_q.device)
