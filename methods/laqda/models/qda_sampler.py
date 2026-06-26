import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        acc = []
        for i in range(self.nway):
            min_index = i * self.qshot
            max_index = (i + 1) * self.qshot - 1
            for j in range(self.kshot):
                count = 0.0
                for z in range(self.k):
                    if min_index <= indices[i][j][z] <= max_index:
                        count += 1
                acc.append(count / float(self.k))
        
        acc_tensor = torch.tensor(acc).mean() if acc else torch.tensor(0.0)
        
        nindices = indices.view(-1, self.k)
        convex_feat = []
        for i in range(nindices.shape[0]):
            convex_feat.append(query_emb.index_select(0, nindices[i]))
        convex_feat = torch.stack(convex_feat)
        
        sampled_data = convex_feat.view(self.nway, self.kshot * self.k, self.dim)
        return sampled_data, acc_tensor
