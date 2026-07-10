import torch
import torch.nn as nn
import torch.nn.functional as F

class LaqdaContrastiveLoss(nn.Module):
    """
    Função de perda principal do LAQDA.
    Combina a perda de classificação (Euclidiana) com a perda contrastiva
    para separar os protótipos de classes diferentes.
    """
    def __init__(self, t: float = 5.0):
        super(LaqdaContrastiveLoss, self).__init__()
        self.t = t

    def forward(self, model_outputs: tuple, target: torch.Tensor):
        prototypes, query_embeddings, acc, original_prototypes, sampled_data = model_outputs
        
        tau = 15.0
        q_norm = F.normalize(query_embeddings, p=2, dim=1)
        p_norm = F.normalize(prototypes, p=2, dim=1)
        
        # Similaridade do Cosseno
        sim = torch.mm(q_norm, p_norm.t())
        
        # Distância como o negativo da similaridade escalonada
        dists = -sim * tau
        
        # CROSS-ENTROPY FIX: 
        # Ao invés de apenas puxar a query para o protótipo correto (o que causa mode collapse),
        # usamos Cross-Entropy nas distâncias negativas para afastar de protótipos incorretos.
        logits = -dists
        target_indices = torch.argmax(target, dim=1)
        ce_loss_fn = nn.CrossEntropyLoss()
        loss = ce_loss_fn(logits, target_indices)
        
        num_classes = prototypes.shape[0]
        if num_classes > 1:
            # Para a perda contrastiva, mantemos a lógica de afastar as classes
            # usando a própria similaridade. p_sim é a matriz C x C
            p_sim = torch.mm(p_norm, p_norm.t())
            p_dists = -p_sim * tau
            
            mask = ~torch.eye(num_classes, dtype=torch.bool, device=p_dists.device)
            contrastive_loss = torch.exp(-p_dists[mask] / self.t).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=dists.device)
            
        total_loss = loss + contrastive_loss
        
        with torch.no_grad():
            preds = torch.argmin(dists, dim=1)
            targets = torch.argmax(target, dim=1)
            
            correct = (preds == targets).sum().float()
            accuracy = correct / len(targets)
            
            classes = torch.unique(targets)
            precisions, recalls, f1s = [], [], []
            for c in classes:
                tp = ((preds == c) & (targets == c)).sum().float()
                fp = ((preds == c) & (targets != c)).sum().float()
                fn = ((preds != c) & (targets == c)).sum().float()
                
                p = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=dists.device)
                r = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=dists.device)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else torch.tensor(0.0, device=dists.device)
                
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)
                
            macro_p = torch.stack(precisions).mean() if precisions else torch.tensor(0.0)
            macro_r = torch.stack(recalls).mean() if recalls else torch.tensor(0.0)
            macro_f1 = torch.stack(f1s).mean() if f1s else torch.tensor(0.0)
            
            topk_acc = accuracy 
            auc = torch.tensor(0.0)
            
        return total_loss, macro_p.item(), macro_r.item(), macro_f1.item(), accuracy.item(), auc.item(), topk_acc.item()
