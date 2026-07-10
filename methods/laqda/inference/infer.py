import os
import torch
import json
import zipfile
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from ..modules.laqda_module import LaqdaModule


class LaqdaInferencer:
    """
    Classe para inferência usando o modelo LAQDA em novos conjuntos de teste.
    """
    def __init__(self, model_paths: list, config: dict, device: str = 'cpu'):
        self.model_paths = model_paths
        self.config = config
        self.device = device
        self.models = self._load_models()
        
    def _load_models(self):
        models = []
        for path in self.model_paths:
            print(f"Loading model from: {path}")
            model_cfg = self.config.get('model', {})
            sampler_cfg = self.config.get('sampler', {})
            
            from ..utils.config_loader import load_config
            global_config_path = 'configs/model_encoder_config.yaml'
            if os.path.exists(global_config_path):
                global_config = load_config(global_config_path)
                lang = global_config.get('model', {}).get('active_language', 'pt')
                global_model_name = global_config.get('model', {}).get(f'encoder_name_{lang}', 'bert-base-uncased')
            else:
                global_model_name = 'bert-base-uncased'
            
            m = LaqdaModule(
                model_name=global_model_name,
                nway=sampler_cfg.get('nway', 2),
                kshot=sampler_cfg.get('kshot', 5),
                qshot=sampler_cfg.get('qshot', 25),
                la=model_cfg.get('la', 1),
                num_freeze=model_cfg.get('num_freeze', 6),
                k=model_cfg.get('k', 5)
            )
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.to(self.device)
            m.eval()
            models.append(m)
        return models

    def prepare_support_set(self, train_file: str, labels_dict: dict, kshot: int):
        class_buckets = defaultdict(list)
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                lbl = item.get('label') or item.get('class_name')
                if lbl in labels_dict:
                    sentence = item.get('sentence') or item.get('text')
                    class_buckets[lbl].append(sentence)
                    
        fixed_support_text = []
        id2label = {v: k for k, v in labels_dict.items()}
        
        import random
        random.seed(self.config.get('hardware', {}).get('seed', 42))
        
        for idx in range(len(labels_dict)):
            label_name = id2label[idx]
            sentences = class_buckets[label_name]
            
            if not sentences:
                print(f"WARNING: No sentences for class {label_name}")
                continue

            if len(sentences) >= kshot:
                selected = random.sample(sentences, kshot)
            else:
                selected = sentences * (kshot // len(sentences)) + sentences[:kshot % len(sentences)]
                
            fixed_support_text.extend(selected)
            
        return fixed_support_text

    def predict_ensemble(self, dataset, support_text: list, labels_dict: dict, batch_size=32):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]
        
        # O DataLoader padronizado retorna (texto, label_id). Pegaremos só o texto.
        dataloader = DataLoader([x['text'] for x in dataset], batch_size=batch_size, shuffle=False)
        
        all_models_preds = []
        
        for m_idx, model in enumerate(self.models):
            print(f"Inferindo com modelo {m_idx+1}/{len(self.models)}...")
            fold_predictions = []
            with torch.no_grad():
                for batch_queries in tqdm(dataloader):
                    input_text = support_text + list(batch_queries)
                    
                    model_outputs = model(input_text, label_text_list)
                    prototypes = model_outputs[0]
                    query_embeddings = model_outputs[1]
                    
                    tau = 15.0
                    q_norm = F.normalize(query_embeddings, p=2, dim=1)
                    p_norm = F.normalize(prototypes, p=2, dim=1)
                    sim = torch.mm(q_norm, p_norm.t())
                    dists = -sim * tau
                    preds_idx = torch.argmin(dists, dim=1).cpu().numpy()
                    
                    for idx in preds_idx:
                        fold_predictions.append(id2label[idx])
            all_models_preds.append(fold_predictions)

        final_preds = []
        num_samples = len(all_models_preds[0])
        for i in range(num_samples):
            votes = [model_p[i] for model_p in all_models_preds]
            winner = Counter(votes).most_common(1)[0][0]
            final_preds.append(winner)
            
        return final_preds

    # =========================================================================
    # MÉTODO ORIGINAL: Score = -min_dist (similaridade cosseno ao protótipo mais próximo)
    # =========================================================================
    def evaluate_ood(self, dataset, support_text: list, labels_dict: dict, batch_size=32, save_dir='./results'):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = [item['class_id'] for item in batch]
            return texts, torch.tensor(labels, dtype=torch.long)
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        all_dists = []
        all_targets = []
        
        model = self.models[0] 
        print("Avaliando OOD metrics (test_final)...")
        
        with torch.no_grad():
            for texts, labels in tqdm(dataloader):
                input_text = support_text + list(texts)
                kshot = self.config.get('sampler', {}).get('kshot', 5)
                model_outputs = model(input_text, label_text_list, kshot=kshot)
                prototypes = model_outputs[0]
                query_embeddings = model_outputs[1]
                
                tau = 15.0
                q_norm = F.normalize(query_embeddings, p=2, dim=1)
                p_norm = F.normalize(prototypes, p=2, dim=1)
                sim = torch.mm(q_norm, p_norm.t())
                dists = -sim * tau
                
                all_dists.append(dists.cpu())
                all_targets.append(labels.cpu())
                
        all_dists_t = torch.cat(all_dists)
        all_targets_t = torch.cat(all_targets)
        
        # Confidência OOD: Menor distância para um protótipo significa maior confiança de ser ID
        # Invertemos o sinal para o reporter (maior = mais confiante)
        min_dists, preds = torch.min(all_dists_t, dim=1)
        confidences = -min_dists
        
        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]
        
        # Verificar se modelo tem limiar SGR travado (LAQDA com SGR)
        sgr_extras = {}
        if hasattr(model, 'sgr_threshold') and model.sgr_threshold.item() != -float('inf'):
            sgr_th = model.sgr_threshold.item()
            # O SGR é calculado com a confiança ID
            abstained = (confidences < sgr_th)
            abstention_rate = abstained.float().mean().item()
            
            # Precisão apenas sobre as predições aceitas (onde abstained == False)
            accepted_mask = ~abstained
            if accepted_mask.sum() > 0:
                acc_accepted = (preds[accepted_mask] == all_targets_t[accepted_mask]).float().mean().item()
            else:
                acc_accepted = 0.0
                
            sgr_extras = {
                "sgr_applied_threshold": sgr_th,
                "sgr_abstention_rate": abstention_rate,
                "sgr_accepted_accuracy": acc_accepted
            }
        
        from methods.metrics.reporter import MetricsReporter
        reporter = MetricsReporter(save_dir=save_dir)
        reporter.generate_report(
            confidences=confidences,
            preds=preds,
            targets=all_targets_t,
            id_scores=id_scores,
            ood_scores=ood_scores,
            model=model,
            prefix="test_final",
            **sgr_extras
        )
        print(f"Relatório test_final_metrics_report.json salvo em {save_dir}")

    # =========================================================================
    # ESTRATÉGIA 1: Distance Margin Score
    # Score = second_min_dist - min_dist
    # Quanto maior a margem entre o 1º e 2º protótipo mais próximo, maior a
    # confiança: o modelo está claramente "preferindo" uma única classe.
    # =========================================================================
    def evaluate_ood_margin(self, dataset, support_text: list, labels_dict: dict, batch_size=32, save_dir='./results'):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = [item['class_id'] for item in batch]
            return texts, torch.tensor(labels, dtype=torch.long)
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        all_sorted_dists = []
        all_sorted_idx = []
        all_targets = []
        
        model = self.models[0]
        print("Avaliando OOD metrics — Estratégia: Distance Margin (test_final_margin)...")
        
        with torch.no_grad():
            for texts, labels in tqdm(dataloader):
                input_text = support_text + list(texts)
                kshot = self.config.get('sampler', {}).get('kshot', 5)
                model_outputs = model(input_text, label_text_list, kshot=kshot)
                prototypes = model_outputs[0]
                query_embeddings = model_outputs[1]
                
                tau = 15.0
                q_norm = F.normalize(query_embeddings, p=2, dim=1)
                p_norm = F.normalize(prototypes, p=2, dim=1)
                sim = torch.mm(q_norm, p_norm.t())
                dists = -sim * tau  # menor dist = maior similaridade

                # Ordena distâncias em ordem crescente (1º = mais próximo)
                sorted_dists, sorted_idx = torch.sort(dists, dim=1)
                all_sorted_dists.append(sorted_dists.cpu())
                all_sorted_idx.append(sorted_idx.cpu())
                all_targets.append(labels.cpu())

        all_sorted_dists_t = torch.cat(all_sorted_dists)
        all_sorted_idx_t = torch.cat(all_sorted_idx)
        all_targets_t = torch.cat(all_targets)

        min_dists = all_sorted_dists_t[:, 0]
        preds = all_sorted_idx_t[:, 0]

        if all_sorted_dists_t.shape[1] >= 2:
            second_dists = all_sorted_dists_t[:, 1]
            # Margem: quanto maior, mais "isolada" está a predição no espaço de embeddings
            # Usamos o negativo da dist mínima + a margem para manter escala consistente
            confidences = second_dists - min_dists
        else:
            # Fallback para caso binário com apenas 1 classe (edge case)
            confidences = -min_dists

        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]

        from methods.metrics.reporter import MetricsReporter
        reporter = MetricsReporter(save_dir=save_dir)
        reporter.generate_report(
            confidences=confidences,
            preds=preds,
            targets=all_targets_t,
            id_scores=id_scores,
            ood_scores=ood_scores,
            model=model,
            prefix="test_final_margin",
        )
        print(f"Relatório test_final_margin_metrics_report.json salvo em {save_dir}")

    # =========================================================================
    # ESTRATÉGIA 2: Monte Carlo Dropout (MC Dropout)
    # Score = mean_similarity - variance_across_passes
    # Mantém dropout ativo durante inferência e faz N passes por batch.
    # Alta variância entre passes = modelo incerto (score baixo).
    # Captura incerteza epistêmica (incerteza do modelo, não dos dados).
    # =========================================================================
    def evaluate_ood_mc_dropout(self, dataset, support_text: list, labels_dict: dict,
                                 batch_size=32, save_dir='./results', n_passes: int = 20):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = [item['class_id'] for item in batch]
            return texts, torch.tensor(labels, dtype=torch.long)
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        model = self.models[0]
        
        # Ativar modo treino APENAS no encoder para manter dropout ativo
        # (o sampler transdutivo fica desativado por estar em modo eval no laqda_module)
        model.train()
        
        print(f"Avaliando OOD metrics — Estratégia: MC Dropout ({n_passes} passes) (test_final_mcdropout)...")
        
        all_mean_sims = []
        all_var_sims = []
        all_preds = []
        all_targets = []
        
        kshot = self.config.get('sampler', {}).get('kshot', 5)
        tau = 15.0
        
        with torch.no_grad():
            for texts, labels in tqdm(dataloader):
                input_text = support_text + list(texts)
                
                # Múltiplos passes estocásticos com dropout ativo
                batch_sims = []  # [n_passes, batch_size, n_classes]
                for _ in range(n_passes):
                    model_outputs = model(input_text, label_text_list, kshot=kshot)
                    prototypes = model_outputs[0]
                    query_embeddings = model_outputs[1]
                    
                    q_norm = F.normalize(query_embeddings, p=2, dim=1)
                    p_norm = F.normalize(prototypes, p=2, dim=1)
                    sim = torch.mm(q_norm, p_norm.t())  # [batch_size, n_classes]
                    batch_sims.append(sim.cpu())
                
                batch_sims_t = torch.stack(batch_sims)  # [n_passes, batch_size, n_classes]
                
                # Média e variância por classe ao longo dos N passes
                mean_sim = batch_sims_t.mean(dim=0)   # [batch_size, n_classes]
                var_sim = batch_sims_t.var(dim=0)      # [batch_size, n_classes]
                
                # Predição pela média das similaridades
                preds_idx = torch.argmax(mean_sim, dim=1)
                
                # Score de confiança:
                # - Pega a similaridade média da classe predita (maior = mais confiante)
                # - Subtrai a variância da classe predita (maior variância = menos confiante)
                mean_pred_sim = mean_sim[torch.arange(len(preds_idx)), preds_idx]
                var_pred_sim = var_sim[torch.arange(len(preds_idx)), preds_idx]
                confidence = mean_pred_sim - var_pred_sim
                
                all_mean_sims.append(mean_pred_sim)
                all_var_sims.append(var_pred_sim)
                all_preds.append(preds_idx)
                all_targets.append(labels.cpu())
        
        # Voltar para modo eval após passes MC
        model.eval()
        
        all_preds_t = torch.cat(all_preds)
        all_targets_t = torch.cat(all_targets)
        mean_sims_t = torch.cat(all_mean_sims)
        var_sims_t = torch.cat(all_var_sims)
        confidences = mean_sims_t - var_sims_t
        
        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]
        
        from methods.metrics.reporter import MetricsReporter
        reporter = MetricsReporter(save_dir=save_dir)
        reporter.generate_report(
            confidences=confidences,
            preds=all_preds_t,
            targets=all_targets_t,
            id_scores=id_scores,
            ood_scores=ood_scores,
            model=model,
            prefix="test_final_mcdropout",
            mc_n_passes=n_passes,
        )
        print(f"Relatório test_final_mcdropout_metrics_report.json salvo em {save_dir}")

    # =========================================================================
    # ESTRATÉGIA 3: Temperature Scaling
    # Aprende o tau (temperatura) ótimo no val set minimizando o NLL.
    # Uma temperatura bem calibrada faz softmax(sim / tau) refletir a real
    # probabilidade de acerto, tornando o SGR muito mais eficaz.
    # =========================================================================
    def evaluate_ood_temp_scaling(self, dataset, support_text: list, labels_dict: dict,
                                   val_dataset=None, batch_size=32, save_dir='./results'):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = [item['class_id'] for item in batch]
            return texts, torch.tensor(labels, dtype=torch.long)

        model = self.models[0]
        kshot = self.config.get('sampler', {}).get('kshot', 5)

        # ── Aprender temperatura no validation set ──────────────────────────
        tau_opt = 15.0  # fallback: temperatura padrão

        if val_dataset is not None:
            print("Temperature Scaling: coletando logits do validation set...")
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            val_logits_all = []
            val_targets_all = []
            
            with torch.no_grad():
                for texts, labels in tqdm(val_dataloader, desc="Val pass"):
                    # Apenas amostras ID para calibração
                    id_mask = labels != -1
                    if id_mask.sum() == 0:
                        continue
                    texts_id = [t for t, m in zip(texts, id_mask) if m]
                    labels_id = labels[id_mask]
                    
                    if not texts_id:
                        continue
                    
                    input_text = support_text + list(texts_id)
                    model_outputs = model(input_text, label_text_list, kshot=kshot)
                    prototypes = model_outputs[0]
                    query_embeddings = model_outputs[1]
                    
                    q_norm = F.normalize(query_embeddings, p=2, dim=1)
                    p_norm = F.normalize(prototypes, p=2, dim=1)
                    sim = torch.mm(q_norm, p_norm.t())  # [batch_id, n_classes]
                    
                    val_logits_all.append(sim.cpu())
                    val_targets_all.append(labels_id.cpu())
            
            if val_logits_all:
                val_logits_t = torch.cat(val_logits_all)   # [N_val, n_classes]
                val_targets_t = torch.cat(val_targets_all) # [N_val]
                
                # Filtrar classes que existem no val set (evita -1 residuais)
                valid_mask = (val_targets_t >= 0) & (val_targets_t < val_logits_t.shape[1])
                val_logits_t = val_logits_t[valid_mask]
                val_targets_t = val_targets_t[valid_mask]
                
                if len(val_targets_t) > 0:
                    from scipy.optimize import minimize_scalar
                    
                    def nll_loss(log_tau):
                        tau = float(np.exp(log_tau))
                        scaled_logits = val_logits_t * tau
                        log_probs = F.log_softmax(scaled_logits, dim=1)
                        nll = F.nll_loss(log_probs, val_targets_t)
                        return nll.item()
                    
                    # Busca no espaço log para evitar tau negativo
                    result = minimize_scalar(nll_loss, bounds=(np.log(0.1), np.log(100.0)), method='bounded')
                    tau_opt = float(np.exp(result.x))
                    print(f"Temperature Scaling: tau ótimo aprendido = {tau_opt:.4f} (original: 15.0)")
                else:
                    print("Temperature Scaling: validation set sem amostras ID válidas. Usando tau=15.0")
            else:
                print("Temperature Scaling: validation set vazio. Usando tau=15.0")
        else:
            print("Temperature Scaling: validation set não fornecido. Usando tau=15.0 (sem calibração)")

        # ── Inferência no test set com tau calibrado ─────────────────────────
        print(f"Avaliando OOD metrics — Estratégia: Temperature Scaling tau={tau_opt:.4f} (test_final_tempscale)...")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        all_dists = []
        all_targets = []
        
        with torch.no_grad():
            for texts, labels in tqdm(dataloader):
                input_text = support_text + list(texts)
                model_outputs = model(input_text, label_text_list, kshot=kshot)
                prototypes = model_outputs[0]
                query_embeddings = model_outputs[1]
                
                q_norm = F.normalize(query_embeddings, p=2, dim=1)
                p_norm = F.normalize(prototypes, p=2, dim=1)
                sim = torch.mm(q_norm, p_norm.t())
                
                # Aplicar temperatura calibrada
                dists = -sim * tau_opt

                all_dists.append(dists.cpu())
                all_targets.append(labels.cpu())

        all_dists_t = torch.cat(all_dists)
        all_targets_t = torch.cat(all_targets)

        # Confiança via softmax calibrado (prob da classe predita)
        probs = F.softmax(-all_dists_t, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]

        from methods.metrics.reporter import MetricsReporter
        reporter = MetricsReporter(save_dir=save_dir)
        reporter.generate_report(
            confidences=confidences,
            preds=preds,
            targets=all_targets_t,
            id_scores=id_scores,
            ood_scores=ood_scores,
            model=model,
            prefix="test_final_tempscale",
            temp_scaling_tau=tau_opt,
        )
        print(f"Relatório test_final_tempscale_metrics_report.json salvo em {save_dir}")

    # =========================================================================
    # ESTRATÉGIA 4: X-Mahalanobis Feature Mixing
    # Usa o Support Set para estimar as matrizes de covariância (pesos) e
    # Mahalanobis global, adaptando o artigo para Few-Shot/LAQDA.
    # =========================================================================
    def evaluate_ood_xmahalanobis(self, dataset, support_text: list, labels_dict: dict,
                                   batch_size=32, save_dir='./results'):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = [item['class_id'] for item in batch]
            return texts, torch.tensor(labels, dtype=torch.long)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        model = self.models[0]
        kshot = self.config.get('sampler', {}).get('kshot', 5)
        
        print("X-Mahalanobis: Extraindo features do Support Set para calibrar pesos e covariância...")
        support_all_hidden = []
        support_labels = []
        
        with torch.no_grad():
            n_classes = len(labels_dict)
            sup_labels = []
            for c in range(n_classes):
                sup_labels.extend([c] * kshot)
            
            from torch.utils.data import DataLoader as TorchDataLoader
            sup_dataset = list(zip(support_text, sup_labels))
            sup_loader = TorchDataLoader(sup_dataset, batch_size=batch_size, shuffle=False)
            
            for texts, labels in sup_loader:
                _, _, _, _, _, all_hidden = model(texts, label_text_list, kshot=kshot, output_hidden_states=True)
                stacked_hidden = torch.stack(all_hidden, dim=0).cpu()
                support_all_hidden.append(stacked_hidden)
                support_labels.append(labels)
                
        support_all_hidden = torch.cat(support_all_hidden, dim=1)
        support_labels = torch.cat(support_labels)
        
        num_layers, total_support, hidden_size = support_all_hidden.shape
        
        # Calcular pesos por camada: alpha^l = Tr( (A^l)^T A^l )
        alphas = []
        for l in range(num_layers):
            layer_feats = support_all_hidden[l]
            mean_feat = layer_feats.mean(dim=0, keepdim=True)
            centered = layer_feats - mean_feat
            variance = (centered ** 2).sum().item()
            alphas.append(variance)
            
        alphas = np.array(alphas)
        alphas = alphas / alphas.sum()
        print(f"Pesos X-Mahalanobis por camada: {np.round(alphas, 4)}")
        
        alphas_t = torch.tensor(alphas, dtype=torch.float32).unsqueeze(1).unsqueeze(2) # (num_layers, 1, 1)
        
        # Computar features misturadas para o Support Set: Phi(x) = sum(alpha^l * x^l)
        support_mixed = (support_all_hidden * alphas_t).sum(dim=0)
        
        # Estimar mu_c e Sigma globais no espaço misturado
        class_means = []
        for c in range(n_classes):
            c_feats = support_mixed[support_labels == c]
            class_means.append(c_feats.mean(dim=0))
        class_means = torch.stack(class_means)
        
        centered_mixed = []
        for i in range(total_support):
            c = support_labels[i].item()
            centered_mixed.append(support_mixed[i] - class_means[c])
        centered_mixed = torch.stack(centered_mixed)
        
        cov = torch.matmul(centered_mixed.t(), centered_mixed) / total_support
        
        jitter = 1e-6 * torch.eye(hidden_size)
        cov = cov + jitter
        inv_cov = torch.linalg.inv(cov)
        
        device = next(model.parameters()).device
        alphas_t = alphas_t.to(device).squeeze(2) # (num_layers, 1)
        class_means = class_means.to(device)
        inv_cov = inv_cov.to(device)
        
        print(f"Avaliando OOD metrics — Estratégia: X-Mahalanobis (test_final_xmaha)...")
        all_dists = []
        all_targets = []
        
        with torch.no_grad():
            for texts, labels in tqdm(dataloader):
                _, _, _, _, _, all_hidden = model(texts, label_text_list, kshot=kshot, output_hidden_states=True)
                stacked_hidden = torch.stack(all_hidden, dim=0)
                
                query_mixed = (stacked_hidden * alphas_t.unsqueeze(2)).sum(dim=0)
                
                batch_size_cur = query_mixed.shape[0]
                dists = torch.zeros(batch_size_cur, n_classes, device=device)
                for c in range(n_classes):
                    diff = query_mixed - class_means[c].unsqueeze(0)
                    left = torch.matmul(diff, inv_cov)
                    mahalanobis_dist = (left * diff).sum(dim=1)
                    dists[:, c] = mahalanobis_dist
                    
                all_dists.append(dists.cpu())
                all_targets.append(labels.cpu())
                
        all_dists_t = torch.cat(all_dists)
        all_targets_t = torch.cat(all_targets)
        
        min_dists, preds = torch.min(all_dists_t, dim=1)
        confidences = -min_dists
        
        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]
        
        sgr_extras = {}
        if hasattr(model, 'sgr_threshold') and model.sgr_threshold.item() != -float('inf'):
            sgr_th = model.sgr_threshold.item()
            abstained = (confidences < sgr_th)
            abstention_rate = abstained.float().mean().item()
            accepted_mask = ~abstained
            acc_accepted = (preds[accepted_mask] == all_targets_t[accepted_mask]).float().mean().item() if accepted_mask.sum() > 0 else 0.0
            sgr_extras = {
                "sgr_applied_threshold": sgr_th,
                "sgr_abstention_rate": abstention_rate,
                "sgr_accepted_accuracy": acc_accepted
            }
            
        from methods.metrics.reporter import MetricsReporter
        reporter = MetricsReporter(save_dir=save_dir)
        reporter.generate_report(
            confidences=confidences,
            preds=preds,
            targets=all_targets_t,
            id_scores=id_scores,
            ood_scores=ood_scores,
            model=model,
            prefix="test_final_xmaha",
            **sgr_extras
        )
        print(f"Relatório test_final_xmaha_metrics_report.json salvo em {save_dir}")
