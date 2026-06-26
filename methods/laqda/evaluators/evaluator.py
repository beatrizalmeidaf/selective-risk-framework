import torch
import numpy as np
from tqdm import tqdm

class LaqdaEvaluator:
    """
    Avalia o modelo LAQDA usando o dataloader episódico.
    """
    def __init__(self, model, loss_fn, device: str, config: dict = None):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.config = config or {}

    def evaluate(self, dataloader, labels_dict: dict, epoch: int = 0):
        self.model.eval()
        batch_loss, batch_acc, batch_f1 = [], [], []
        id2label = {idx: original_label for original_label, idx in labels_dict.items()}

        all_dists = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc=f"Eval Ep {epoch}"):
                support_set, query_set, episode_internal_ids = batch
                label_text = [id2label.get(int(el), str(el)) for el in episode_internal_ids]
                
                text, one_hot_labels = self._format_batch(support_set, query_set, episode_internal_ids, labels_dict)
                one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float).to(self.device)

                if (one_hot_labels_tensor == -1).any():
                    continue

                query_labels_tensor = one_hot_labels_tensor[len(support_set):]
                try:
                    model_outputs = self.model(text, label_text)
                    loss, p, r, f1, acc, auc, topk_acc = self.loss_fn(model_outputs, query_labels_tensor)
                    batch_loss.append(loss.item())
                    batch_acc.append(acc)
                    batch_f1.append(f1)
                    
                    # Accumulate for global metrics
                    prototypes, query_embeddings, _, _, _ = model_outputs
                    dists = torch.pow(query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0), 2).sum(2)
                    all_dists.append(dists.cpu())
                    all_targets.append(torch.argmax(query_labels_tensor, dim=1).cpu())
                    
                except Exception as e:
                    print(f"Eval Error: {e}")
                    continue

        avg_loss = np.mean(batch_loss) if batch_loss else 0
        avg_acc = np.mean(batch_acc) if batch_acc else 0
        avg_f1 = np.mean(batch_f1) if batch_f1 else 0
        
        # Padrão Ouro de Avaliação
        if all_dists:
            all_dists_t = torch.cat(all_dists)
            all_targets_t = torch.cat(all_targets)
            
            # No LAQDA, menores distâncias = maior confiança. 
            # Invertemos o sinal para o reporter
            confidences, preds = torch.max(-all_dists_t, dim=1)
            
            from methods.metrics.reporter import MetricsReporter
            reporter = MetricsReporter()
            report = reporter.generate_report(
                confidences=confidences,
                preds=preds,
                targets=all_targets_t,
                model=self.model,
                prefix=f"laqda_val_ep_{epoch}"
            )
            
            use_sgr = self.config.get('metrics', {}).get('use_sgr', False)
            if use_sgr:
                # Treinamento do Limiar de Risco SGR do LAQDA (Target Risk = 5%)
                from methods.sgr.sgr import SGRController
                sgr = SGRController(delta=0.001)
                best_theta, _, _ = sgr.fit(confidences, preds, all_targets_t, r_star=0.05)
                self.model.sgr_threshold.fill_(best_theta)
                print(f"LAQDA SGR Threshold (Risco 5%) travado em: {best_theta:.4f}")
        
        return avg_loss, avg_acc, avg_f1

    def _format_batch(self, support_set, query_set, episode_internal_ids, labels_dict):
        text, original_labels = [], []
        for x in support_set:
            text.append(x["text"])
            original_labels.append(x["label"])
        for x in query_set:
            text.append(x["text"])
            original_labels.append(x["label"])
            
        one_hot_labels = []
        num_classes_in_episode = len(episode_internal_ids)
        internal_id_to_episode_index = {internal_id: idx for idx, internal_id in enumerate(episode_internal_ids)}

        for orig_label in original_labels:
            internal_id = labels_dict.get(orig_label)
            if internal_id is None or internal_id not in internal_id_to_episode_index:
                one_hot = [-1] * num_classes_in_episode
            else:
                episode_index = internal_id_to_episode_index[internal_id]
                one_hot = [0] * num_classes_in_episode
                one_hot[episode_index] = 1
            one_hot_labels.append(one_hot)

        return text, one_hot_labels
