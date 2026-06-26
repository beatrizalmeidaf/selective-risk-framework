import torch
import torch.nn as nn
from ..models.encoder import LabelAwareEncoder
from ..models.qda_sampler import TransductiveQDASampler

class LaqdaModule(nn.Module):
    """
    Módulo orquestrador do LAQDA.
    Conecta o Encoder Label-Aware ao Sampler Transdutivo QDA.
    """
    def __init__(self, model_name: str, nway: int, kshot: int, qshot: int, la: int = 1, num_freeze: int = 6, k: int = 5):
        super(LaqdaModule, self).__init__()
        self.nway = nway
        self.kshot = kshot
        
        self.encoder = LabelAwareEncoder(model_name, la=la, num_freeze=num_freeze)
        self.sampler = TransductiveQDASampler(self.encoder.hidden_size, nway, kshot, qshot, k)
        
        # Buffer para guardar o Threshold ótimo de rejeição aprendido pelo SGR
        self.register_buffer('sgr_threshold', torch.tensor(-float('inf')))

    def forward(self, text: list, label_texts: list):
        support_size = self.nway * self.kshot
        
        text_embedding = self.encoder(text, label_texts)
        
        support_emb = text_embedding[:support_size]
        query_emb = text_embedding[support_size:]
        
        c_prototypes = support_emb.view(self.nway, max(1, self.kshot), -1)
        original_prototypes = c_prototypes.mean(dim=1)
        
        sampled_data, acc = self.sampler(support_emb, query_emb)
        
        prototypes_data = torch.cat((c_prototypes, sampled_data), dim=1)
        prototypes = torch.mean(prototypes_data, dim=1)
        
        return prototypes, query_emb, acc, original_prototypes, sampled_data
