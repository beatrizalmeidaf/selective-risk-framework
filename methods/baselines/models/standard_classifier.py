import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BaselineClassifier(nn.Module):
    """
    Classificador Genérico de Baseline.
    Usa um Encoder (ex: BERT) e uma camada linear (Cross-Entropy).
    Diferente do LAQDA, aqui retornamos as features (penúltima camada)
    e as logits (última camada) isoladas para servir de input aos scorers OOD.
    """
    def __init__(self, model_name: str, num_classes: int, num_freeze: int = 6, dropout: float = 0.1):
        super(BaselineClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Obter o tamanho do embedding dinamicamente
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes)
        )
        
        if num_freeze > 0:
            total_layers = getattr(self.encoder.config, 'num_hidden_layers', 12)
            self._freeze_layers(min(num_freeze, total_layers), total_layers)

    def _freeze_layers(self, num_freeze: int, total_layers: int):
        unfreeze_layers = []
        if num_freeze < total_layers:
            unfreeze_layers.append("pooler")
            
        for i in range(num_freeze, total_layers):
            unfreeze_layers.append(f"layer.{i}")
            
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        
    def train(self, mode=True):
        super().train(mode)
        # Se o encoder estiver completamente congelado, forçamos o modo eval
        # para desativar os dropouts internos do BERT e estabilizar a extração de features
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()

    def forward(self, text: list):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        outputs = self.encoder(**inputs)
        
        # Ignoramos o pooler_output (pois é treinado para NSP e distorce o espaço quando congelado)
        # Usamos o embedding direto do token [CLS]
        features = outputs.last_hidden_state[:, 0, :]
            
        logits = self.classifier(features)
        
        return features, logits
