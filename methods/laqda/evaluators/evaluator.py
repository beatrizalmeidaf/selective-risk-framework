import math
import torch
import numpy as np
from tqdm import tqdm

from ..scoring import compute_prototype_scores

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

                query_labels_tensor = one_hot_labels_tensor[len(support_set):]
                
                # Pular batches com labels -1 (OOD). Validação é estritamente ID.
                # (Protocolo: OOD não pode ser visto durante validação --- standard da literatura)
                if (one_hot_labels_tensor == -1).any():
                    continue
                
                try:
                    model_outputs = self.model(text, label_text)
                    prototypes, query_embeddings, _, _, _ = model_outputs
                    loss, p, r, f1, acc, auc, topk_acc = self.loss_fn(model_outputs, query_labels_tensor)
                    batch_loss.append(loss.item())
                    batch_acc.append(acc)
                    batch_f1.append(f1)
                    
                    # Accumulate for global metrics (apenas ID).
                    # CRÍTICO: usa o MESMO score canônico da inferência (cosseno x tau,
                    # methods/laqda/scoring.py). Se a métrica/escala divergir daqui,
                    # o threshold SGR calibrado abaixo NÃO transfere para o teste.
                    dists, _, _, _ = compute_prototype_scores(query_embeddings, prototypes)
                    all_dists.append(dists.cpu())
                    all_targets.append(torch.argmax(query_labels_tensor, dim=1).cpu())
                    
                except Exception as e:
                    is_oom = "out of memory" in str(e).lower()
                    print(f"Eval Error{' [OOM]' if is_oom else ''}: {e}")
                    if is_oom:
                        torch.cuda.empty_cache()
                    continue

        if not batch_loss:
            # Sem isso, val_loss cai pra 0 (np.mean condicional) e o trainer
            # interpreta erroneamente como "melhor loss de validação já visto",
            # travando para sempre um checkpoint que nunca foi de fato avaliado.
            raise RuntimeError(
                f"Época {epoch}: nenhum batch de validação foi bem-sucedido. "
                f"Abortando em vez de reportar val_loss=0 (falso 'melhor modelo')."
            )

        avg_loss = np.mean(batch_loss) if batch_loss else 0
        avg_acc = np.mean(batch_acc) if batch_acc else 0
        avg_f1 = np.mean(batch_f1) if batch_f1 else 0
        
        # Padrão Ouro de Avaliação
        if all_dists:
            all_dists_t = torch.cat(all_dists)
            all_targets_t = torch.cat(all_targets)
            
            # Score DEFAULT (-min_dist em escala cosseno x tau) — idêntico ao score
            # sobre o qual o threshold é aplicado na inferência (estratégia DEFAULT
            # de infer.py). O SGR é calibrado NESTA escala e somente nela.
            min_dists, preds = torch.min(all_dists_t, dim=1)
            confidences = -min_dists
            
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
                # Treinamento do Limiar de Risco SGR do LAQDA (Target Risk = 10%)
                from methods.sgr.sgr import SGRController
                sgr = SGRController(delta=0.05)
                r_star = self.config.get('metrics', {}).get('sgr_r_star', 0.10)
                best_theta, _, _ = sgr.fit(confidences, preds, all_targets_t, r_star=r_star)
                # sgr.fit retorna theta=+inf quando NENHUM threshold atinge o risco
                # alvo (bound acima do target) — não é um limiar válido. Gravar +inf
                # no buffer fazia a inferência (evaluate_ood) tratá-lo como calibrado
                # (só excluía o sentinela -inf), gerando confidences < inf = 100%
                # abstenção e um valor infinito que quebrava o relatório na saída.
                # Mantemos -inf (mesmo sentinela do default) para sinalizar "sem
                # threshold viável", igual ao restante do código já assume.
                self.model.sgr_threshold_10.fill_(best_theta if math.isfinite(best_theta) else -float('inf'))
                print(f"LAQDA SGR Threshold (Risco {r_star*100}%) travado em: {best_theta:.4f}")

                # Exibir coverage para outros níveis de risco (análise few-shot)
                for r_test in [0.15, 0.20, 0.25]:
                    t_val, b_val, c_val = sgr.fit(confidences, preds, all_targets_t, r_star=r_test)
                    print(f"  [Info] SGR alcançável p/ risco {r_test*100}% -> Cobertura: {c_val:.4f}")
                    t_val = t_val if math.isfinite(t_val) else -float('inf')
                    if r_test == 0.15 and hasattr(self.model, 'sgr_threshold_15'): self.model.sgr_threshold_15.fill_(t_val)
                    if r_test == 0.20 and hasattr(self.model, 'sgr_threshold_20'): self.model.sgr_threshold_20.fill_(t_val)
                    if r_test == 0.25 and hasattr(self.model, 'sgr_threshold_25'): self.model.sgr_threshold_25.fill_(t_val)
        
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
