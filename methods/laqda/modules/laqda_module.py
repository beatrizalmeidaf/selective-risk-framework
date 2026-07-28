import torch
import torch.nn as nn
from ..models.encoder import LabelAwareEncoder
from ..models.qda_sampler import TransductiveQDASampler, sinkhorn_prototype_reestimation

class LaqdaModule(nn.Module):
    """
    Módulo orquestrador do LAQDA.
    Conecta o Encoder Label-Aware ao Sampler Transdutivo QDA.
    """
    def __init__(self, model_name: str, nway: int, kshot: int, qshot: int, la: int = 1, num_freeze: int = 6, k: int = 5,
                 use_transductive_sampler: bool = True, sampler_mode: str = "knn"):
        super(LaqdaModule, self).__init__()
        self.nway = nway
        self.kshot = kshot
        # Ablação: com False, o protótipo de treino fica na média pura do suporte
        # (sem o aumento por vizinhos mais próximos da query feito pelo sampler).
        self.use_transductive_sampler = use_transductive_sampler
        # "knn" (padrão, atribuição rígida por rho-vizinhos) ou "sinkhorn"
        # (reestimação via OT entrópico, fiel ao LAQDA original — só afeta treino).
        self.sampler_mode = sampler_mode

        self.encoder = LabelAwareEncoder(model_name, la=la, num_freeze=num_freeze)
        self.sampler = TransductiveQDASampler(self.encoder.hidden_size, nway, kshot, qshot, k)
        
        # Buffers para guardar os Thresholds ótimos de rejeição aprendidos pelo SGR
        self.register_buffer('sgr_threshold_10', torch.tensor(-float('inf'))) # 10%
        self.register_buffer('sgr_threshold_15', torch.tensor(-float('inf')))
        self.register_buffer('sgr_threshold_20', torch.tensor(-float('inf')))
        self.register_buffer('sgr_threshold_25', torch.tensor(-float('inf')))

    def forward(self, text: list, label_texts: list, kshot: int = None, output_hidden_states: bool = False):
        # Durante inferência no dataset completo, label_texts contém TODAS as classes.
        current_nway = len(label_texts)
        current_kshot = kshot if kshot is not None else self.kshot
        
        support_size = current_nway * current_kshot
        
        if output_hidden_states:
            text_embedding, all_hidden_states = self.encoder(text, label_texts, output_hidden_states=True)
        else:
            text_embedding = self.encoder(text, label_texts)
            all_hidden_states = None
        
        support_emb = text_embedding[:support_size]
        query_emb = text_embedding[support_size:]
        
        c_prototypes = support_emb.view(current_nway, max(1, current_kshot), -1)
        original_prototypes = c_prototypes.mean(dim=1)
        
        # O QDA Sampler assume tarefas episódicas exatas (nway x qshot).
        # Em teste/inferência (self.training == False) sobre o dataset inteiro, nós desativamos
        # para evitar RuntimeError de reshape.
        if self.training and current_nway == self.nway and self.use_transductive_sampler:
            if self.sampler_mode == "sinkhorn":
                prototypes = sinkhorn_prototype_reestimation(
                    original_prototypes, query_emb, rho=self.sampler.k)
                # A métrica diagnóstica de pureza de vizinhança não se aplica a
                # um plano de transporte suave.
                acc = 0.0
                sampled_data = None
            else:
                sampled_data, acc = self.sampler(support_emb, query_emb)
                prototypes_data = torch.cat((c_prototypes, sampled_data), dim=1)
                prototypes = torch.mean(prototypes_data, dim=1)
        else:
            prototypes = original_prototypes
            acc = 0.0
            sampled_data = None
            
        if output_hidden_states:
            return prototypes, query_emb, acc, original_prototypes, sampled_data, all_hidden_states
            
        return prototypes, query_emb, acc, original_prototypes, sampled_data
