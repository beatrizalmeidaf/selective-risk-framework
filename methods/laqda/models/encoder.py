import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

class LabelAwareEncoder(nn.Module):
    """
    Encoder principal do LAQDA (Label-Aware Encoder).
    Integra um modelo Transformer (BGE, RoBERTa, BERT, etc) e adiciona
    a lógica de cross-attention entre a sentença e a representação da classe (Label-Aware).
    """
    def __init__(self, model_name: str, la: int = 1, num_freeze: int = 6):
        super(LabelAwareEncoder, self).__init__()
        self.la = la
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        
        self.hidden_size = config.hidden_size
        self.lin = nn.Linear(self.hidden_size, self.hidden_size)
        
        last_layer_index = config.num_hidden_layers - 1
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            self.attention = copy.deepcopy(self.model.encoder.layer[last_layer_index])
        else:
            self.attention = None 
            if self.la == 1:
                print(f"Warning: Model {model_name} does not have 'encoder.layer'. Disabling Label-Aware logic.")
                self.la = 0
                
        self.drop = nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))
        
        if num_freeze > 0:
            self._freeze_layers(min(num_freeze, config.num_hidden_layers), config.num_hidden_layers)

    def _freeze_layers(self, num_freeze: int, total_layers: int):
        unfreeze_layers = ["pooler"]
        for i in range(num_freeze, total_layers):
            unfreeze_layers.append(f"layer.{i}")
            
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def forward(self, text: list, task_classes: list, output_hidden_states: bool = False):
        tokenizer_output = self.tokenizer(
            text, padding=True, truncation=True, max_length=250, return_tensors='pt'
        ).to(self.model.device)
        
        outputs = self.model(**tokenizer_output, output_hidden_states=output_hidden_states)
        last_hidden_state = outputs.last_hidden_state
        
        # Coleta os estados ocultos de todas as camadas (caso solicitado para X-Mahalanobis)
        all_hidden_states = []
        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            # O BERT retorna (embedding_layer, layer_1, ..., layer_N)
            # Ignoramos a camada de embedding [0] e pegamos apenas o CLS token [:, 0, :]
            all_hidden_states = [layer_out[:, 0, :] for layer_out in outputs.hidden_states[1:]]
        
        if self.la == 0 or self.attention is None:
            final_emb = last_hidden_state[:, 0, :]
            if output_hidden_states:
                # O último layer já está em all_hidden_states[-1], mas para consistência:
                return final_emb, tuple(all_hidden_states)
            return final_emb
            
        task_labels_inputs = self.tokenizer(
            task_classes, return_tensors='pt', padding=True, truncation=True
        ).to(self.model.device)
        
        label_outputs = self.model(**task_labels_inputs)
        label_embedding = label_outputs.last_hidden_state[:, 0, :]
        
        sentence_embeddings = last_hidden_state
        label_embeddings_expanded = label_embedding.unsqueeze(0).expand(sentence_embeddings.shape[0], -1, -1)
        
        connect_embeddings = torch.cat((sentence_embeddings, label_embeddings_expanded), dim=1)
        
        sentence_mask = tokenizer_output['attention_mask']
        label_mask = torch.ones(label_embeddings_expanded.shape[0], label_embeddings_expanded.shape[1], device=self.model.device)
        connect_mask = torch.cat((sentence_mask, label_mask), dim=1)
        
        extended_attention_mask = connect_mask.unsqueeze(1).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        linear_output = self.lin(connect_embeddings)
        dropped_output = self.drop(linear_output)
        
        attention_output = self.attention(dropped_output, extended_attention_mask)[0]
        residual_output = 0.1 * attention_output + 0.9 * connect_embeddings
        
        final_emb = residual_output[:, 0, :]
        
        if output_hidden_states:
            # Adicionamos a camada Label-Aware como a última camada do modelo
            all_hidden_states.append(final_emb)
            return final_emb, tuple(all_hidden_states)
            
        return final_emb
