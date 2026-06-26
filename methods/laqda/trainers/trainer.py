import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from ..evaluators.evaluator import LaqdaEvaluator

class LaqdaTrainer:
    """
    Orquestrador de Treinamento do modelo LAQDA.
    """
    def __init__(self, model, loss_fn, config: dict, device: str = 'cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        
        self.patience = config.get('training', {}).get('patience', 20)
        
        self.evaluator = LaqdaEvaluator(model, loss_fn, device, config)

    def setup_optimizer(self):
        train_cfg = self.config.get('training', {})
        learning_rate = train_cfg.get('learning_rate', 5e-6)
        weight_decay = train_cfg.get('weight_decay', 6.68e-04)
        warmup_steps = train_cfg.get('warmup_steps', 100)
        total_steps = train_cfg.get('epochs', 100) * train_cfg.get('episode_train', 100)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 
             'weight_decay': 0.0}
        ]
        
        has_trainable = any(p.requires_grad for group in optimizer_grouped_parameters for p in group['params'])
        if not has_trainable:
            print("Warning: No trainable parameters found. Optimizer will mock training.")
            self.optimizer = AdamW([torch.nn.Parameter(torch.zeros(1))], lr=learning_rate)
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
            
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def fit(self, tr_dataloader, labels_dict: dict, val_dataloader=None, save_dir: str = './outputs'):
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, 'acc_best_model.pth')
        
        epochs = self.config.get('training', {}).get('epochs', 100)
        
        acc_best_state = None
        best_acc = 0.0
        cycle = 0
        
        id2label = {idx: original_label for original_label, idx in labels_dict.items()}

        for epoch in range(epochs):
            print(f'\n=== Epoch {epoch}/{epochs-1} ===')
            self.model.train()
            
            if cycle >= self.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

            batch_loss, batch_acc, batch_f1 = [], [], []

            for i, batch in tqdm(enumerate(tr_dataloader), total=len(tr_dataloader), desc=f"Train Ep {epoch}"):
                self.optimizer.zero_grad()
                support_set, query_set, episode_internal_ids = batch
                
                label_text = [id2label.get(int(el), str(el)) for el in episode_internal_ids]
                text, one_hot_labels = self.evaluator._format_batch(support_set, query_set, episode_internal_ids, labels_dict)
                one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float).to(self.device)

                if (one_hot_labels_tensor == -1).any():
                    continue

                query_labels_tensor = one_hot_labels_tensor[len(support_set):]
                try:
                    model_outputs = self.model(text, label_text)
                    loss, p, r, f1, acc, auc, topk_acc = self.loss_fn(model_outputs, query_labels_tensor)
                except Exception as e:
                    print(f"Train Error batch {i}: {e}")
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                batch_loss.append(loss.item())
                batch_acc.append(acc)
                batch_f1.append(f1)

            avg_loss = np.mean(batch_loss) if batch_loss else 0
            avg_acc = np.mean(batch_acc) if batch_acc else 0
            avg_f1 = np.mean(batch_f1) if batch_f1 else 0
            
            print(f"Média Train: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, F1={avg_f1:.4f}")

            if val_dataloader:
                val_loss, val_acc, val_f1 = self.evaluator.evaluate(val_dataloader, labels_dict, epoch)
                print(f"Média Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
                
                if val_acc > best_acc:
                    print(f"Nova melhor Acc Validação: {val_acc:.4f}. Salvando...")
                    torch.save(self.model.state_dict(), best_model_path)
                    best_acc = val_acc
                    acc_best_state = copy.deepcopy(self.model.state_dict())
                    cycle = 0
                else:
                    cycle += 1
            else:
                torch.save(self.model.state_dict(), best_model_path)

        if acc_best_state is not None:
            self.model.load_state_dict(acc_best_state)
        elif os.path.exists(best_model_path):
            try:
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            except: pass
                
        return self.model
