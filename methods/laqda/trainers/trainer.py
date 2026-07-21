import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        
        # Otimização H100: BFloat16 para velocidade dobrada
        self.scaler = torch.amp.GradScaler('cuda', enabled=True)

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
            
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        print(f"\n[*] Optimizer iniciado do zero. Learning Rate base: {learning_rate:.2e}")

    def fit(self, tr_dataloader, labels_dict: dict, val_dataloader=None, save_dir: str = './outputs'):
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        
        epochs = self.config.get('training', {}).get('epochs', 100)
        
        acc_best_state = None
        best_val_loss = float('inf')
        cycle = 0
        
        id2label = {idx: original_label for original_label, idx in labels_dict.items()}

        for epoch in range(epochs):
            print(f'\n=== Epoch {epoch}/{epochs-1} ===')
            self.model.train()
            
            if cycle >= self.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

            batch_loss, batch_acc, batch_f1 = [], [], []
            consecutive_failures = 0
            # Falha catastrófica (ex: CUDA OOM) repetida em toda batch travava
            # silenciosamente aqui: o except engolia o erro, a época "terminava"
            # com Loss=0.0000 (lista vazia -> np.mean condicional cai pra 0) e o
            # treino seguia por dezenas de épocas fingindo funcionar, salvando um
            # "best_model" que na prática nunca recebeu um único gradiente real
            # (ver outputs/final_eval/.../MMLU.../kshot_5|10 antes desta correção).
            # Threshold aqui aborta ALTO e CEDO em vez de desperdiçar o job inteiro.
            max_consecutive_failures = 20

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
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        model_outputs = self.model(text, label_text)
                        loss, p, r, f1, acc, auc, topk_acc = self.loss_fn(model_outputs, query_labels_tensor)
                except Exception as e:
                    is_oom = "out of memory" in str(e).lower()
                    print(f"Train Error batch {i}{' [OOM]' if is_oom else ''}: {e}")
                    if is_oom:
                        # Libera blocos cacheados para dar chance ao próximo batch;
                        # sem isso a fragmentação faz TODO batch seguinte falhar também.
                        torch.cuda.empty_cache()
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise RuntimeError(
                            f"{consecutive_failures} batches consecutivos falharam "
                            f"(último erro: {e}). Abortando em vez de completar o "
                            f"treino silenciosamente sem gradientes reais — reduza "
                            f"kshot/qshot/max_query_total ou o batch de encoding."
                        ) from e
                    continue

                consecutive_failures = 0

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                # Scaled Backward Pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_loss.append(loss.item())
                batch_acc.append(acc)
                batch_f1.append(f1)

            if not batch_loss:
                raise RuntimeError(
                    f"Época {epoch}: nenhum batch de treino foi bem-sucedido. "
                    f"O modelo não recebeu nenhum gradiente real nesta época — "
                    f"abortando em vez de salvar um checkpoint não treinado."
                )

            avg_loss = np.mean(batch_loss) if batch_loss else 0
            avg_acc = np.mean(batch_acc) if batch_acc else 0
            avg_f1 = np.mean(batch_f1) if batch_f1 else 0
            
            print(f"Média Train: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, F1={avg_f1:.4f}")

            if val_dataloader:
                val_loss, val_acc, val_f1 = self.evaluator.evaluate(val_dataloader, labels_dict, epoch)
                print(f"Média Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
                
                if val_loss < best_val_loss:
                    print(f"Nova melhor Loss Validação: {val_loss:.4f}. Salvando...")
                    torch.save(self.model.state_dict(), best_model_path)
                    best_val_loss = val_loss
                    acc_best_state = copy.deepcopy(self.model.state_dict())
                    cycle = 0
                else:
                    cycle += 1
                    
                if self.lr_scheduler:
                    self.lr_scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Current LR: {current_lr:.2e}")
            else:
                torch.save(self.model.state_dict(), best_model_path)

        if acc_best_state is not None:
            self.model.load_state_dict(acc_best_state)
        elif os.path.exists(best_model_path):
            try:
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            except: pass
                
        return self.model
