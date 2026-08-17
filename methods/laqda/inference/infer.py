import os
import math
import torch
import json
import zipfile
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from ..modules.laqda_module import LaqdaModule
from ..scoring import compute_prototype_scores


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
            # Mesmo override de train.py: sem a variavel, o valor do YAML e'
            # preservado. Sem isto, treinar com um encoder alternativo produz
            # checkpoints que a inferencia nao consegue carregar.
            _enc_override = os.environ.get('LAQDA_ENCODER')
            if _enc_override:
                global_model_name = _enc_override
            print(f"Encoder (infer): {global_model_name}", flush=True)

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
                # Pular a classe aqui (comportamento antigo) encolhe fixed_support_text
                # abaixo de nway*kshot em silêncio. LaqdaModule.forward fatia
                # text_embedding em suporte/query usando support_size = nway*kshot
                # (calculado a partir de nway = len(label_text_list), não do tamanho
                # real de fixed_support_text) — com 1 classe faltando, esse corte cai
                # 1 posição errado: rouba a 1ª amostra de query pra virar "suporte" da
                # classe vazia e desalinha TODO o resto do batch por 1. Pra kshot=5/10
                # isso ainda quebra alto com RuntimeError de reshape (visível); pra
                # kshot=1 o reshape "funciona" por coincidência de tamanho e o erro só
                # aparece muito depois, num IndexError de mask sem relação aparente —
                # ou pior, sem erro nenhum, com métricas silenciosamente erradas.
                # Falhar alto aqui, na causa raiz, é sempre preferível a essa corrupção
                # silenciosa (ver RulingBRCorpus fold 01 "direito financeiro" / fold 02
                # "direito notarial": 0 exemplos no train.jsonl daquele fold).
                raise ValueError(
                    f"Classe '{label_name}' está em labels_dict (id_classes) mas tem "
                    f"ZERO exemplos em {train_file}. Não é seguro montar o support set "
                    f"faltando uma classe — corrija configs/ood_splits.json (exclua essa "
                    f"classe do id_classes deste fold) ou reveja os dados desse fold."
                )

            if len(sentences) >= kshot:
                selected = random.sample(sentences, kshot)
            else:
                selected = sentences * (kshot // len(sentences)) + sentences[:kshot % len(sentences)]
                
            fixed_support_text.extend(selected)

        return fixed_support_text

    def _fit_and_apply_sgr(self, val_confidences, val_preds, val_targets,
                            test_confidences, test_preds, test_targets, id_mask, ood_mask):
        """
        Calibra thresholds SGR (risco 10/15/20/25%) em confidences/preds/targets de
        VALIDAÇÃO (estritamente ID) e aplica ao teste (ID+OOD). Compartilhado entre
        todas as estratégias de score para não duplicar o loop de calibração em cada
        evaluate_ood_*.

        IMPORTANTE: cada estratégia deve passar confidences calculadas com SEU PRÓPRIO
        score (margin, mcdropout, tempscale, xmaha, ...) tanto no val quanto no teste.
        Um threshold calibrado numa escala de score (ex: cosseno x tau) não transfere
        para outra escala (ex: softmax calibrado, margem, Mahalanobis) — ver o aviso em
        methods/laqda/scoring.py.
        """
        from methods.sgr.sgr import SGRController
        sgr = SGRController(delta=0.05)
        sgr_extras = {}

        # m e' o tamanho do conjunto de calibracao held-out; e' o parametro que
        # governa a folga do bound e precisa ser reportado junto da cobertura.
        sgr_extras['sgr_calib_m'] = int(len(val_confidences))
        for risk_pct, r_star in [(5, 0.05), (10, 0.10), (15, 0.15), (20, 0.20), (25, 0.25)]:
            theta, bound, cov_val = sgr.fit(val_confidences, val_preds, val_targets, r_star=r_star)
            if theta == float('inf'):
                print(f"  [Info] SGR r*={r_star:.2f}: nenhum threshold viável (bound acima do alvo).")
                continue

            abstained = (test_confidences < theta)
            abstention_rate = abstained.float().mean().item()

            accepted_mask = ~abstained
            if accepted_mask.sum() > 0:
                acc_accepted = (test_preds[accepted_mask] == test_targets[accepted_mask]).float().mean().item()
            else:
                acc_accepted = 0.0

            # Decomposição da abstenção: cobertura sobre ID (quanto do serviço é
            # mantido) e rejeição sobre OOD (proteção emergente).
            id_coverage = (~abstained[id_mask]).float().mean().item() if id_mask.sum() > 0 else 0.0
            ood_rejection = abstained[ood_mask].float().mean().item() if ood_mask.sum() > 0 else 0.0

            prefix = "" if risk_pct == 10 else f"_{risk_pct}"
            sgr_extras.update({
                f"sgr_applied_threshold{prefix}": theta,
                f"sgr_val_coverage{prefix}": cov_val,
                f"sgr_abstention_rate{prefix}": abstention_rate,
                f"sgr_accepted_accuracy{prefix}": acc_accepted,
                f"sgr_id_coverage{prefix}": id_coverage,
                f"sgr_ood_rejection_rate{prefix}": ood_rejection
            })
            print(f"  [SGR] r*={r_star:.2f} -> theta={theta:.4f} | "
                  f"cobertura val={cov_val:.4f} | abstenção teste={abstention_rate:.4f}")

        return sgr_extras

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
                    
                    _, preds_t, _, _ = compute_prototype_scores(query_embeddings, prototypes)
                    preds_idx = preds_t.cpu().numpy()
                    
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
    def evaluate_ood(self, dataset, support_text: list, labels_dict: dict, batch_size=32, save_dir='./results', val_dataset=None):
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
                
                # Score canônico compartilhado com a calibração do SGR
                # (methods/laqda/scoring.py) — mesma métrica, normalização e tau.
                dists, _, _, _ = compute_prototype_scores(query_embeddings, prototypes)

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
        
        # id_only_accuracy é calculado centralmente em compute_accuracy_f1
        # (methods/metrics/classification.py) a partir de targets/preds.
        sgr_extras = {}

        # Calibração SGR held-out para o score DEFAULT. Antes, este caminho lia
        # thresholds dos buffers do modelo, que são gravados no fim do treino
        # sobre as queries episódicas de validação -- o mesmo fluxo monitorado
        # pelo early stopping, portanto NÃO disjunto. Aqui calibramos no split
        # de validação (só ID) e aplicamos ao teste sem alteração, que é a
        # condição que a garantia exige.
        if val_dataset is not None:
            print("Default score: calibrando thresholds SGR no validation set (held-out)...")
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            v_dists, v_targets = [], []
            with torch.no_grad():
                for texts, labels in tqdm(val_loader):
                    input_text = support_text + list(texts)
                    kshot = self.config.get('sampler', {}).get('kshot', 5)
                    mo = model(input_text, label_text_list, kshot=kshot)
                    d, _, _, _ = compute_prototype_scores(mo[1], mo[0])
                    v_dists.append(d.cpu()); v_targets.append(labels.cpu())
            v_dists_t = torch.cat(v_dists); v_targets_t = torch.cat(v_targets)
            v_min, v_preds = torch.min(v_dists_t, dim=1)
            v_conf = -v_min
            v_id = v_targets_t != -1
            sgr_extras.update(self._fit_and_apply_sgr(
                v_conf[v_id], v_preds[v_id], v_targets_t[v_id],
                confidences, preds, all_targets_t, id_mask, ood_mask
            ))

        # Verificar se modelo tem limiares SGR travados (LAQDA com SGR)
        thresholds_to_check = [
            (10, 'sgr_threshold_10'),
            (15, 'sgr_threshold_15'),
            (20, 'sgr_threshold_20'),
            (25, 'sgr_threshold_25')
        ]
        
        for risk_pct, attr_name in thresholds_to_check:
            if not hasattr(model, attr_name):
                continue
            sgr_th = getattr(model, attr_name).item()
            # -inf = threshold nunca calibrado (default do buffer); +inf = SGR
            # calibrou e não achou threshold viável no risco alvo (ver
            # evaluator.py, sgr.fit pode retornar theta=inf). Os dois casos são
            # "sem threshold aplicável" — só != -inf deixava o +inf passar como
            # se fosse válido, gerando confidences < inf (100% abstenção) e um
            # relatório com valor infinito que quebrava o print em reporter.py.
            if math.isfinite(sgr_th):
                abstained = (confidences < sgr_th)
                abstention_rate = abstained.float().mean().item()
                
                accepted_mask = ~abstained
                if accepted_mask.sum() > 0:
                    acc_accepted = (preds[accepted_mask] == all_targets_t[accepted_mask]).float().mean().item()
                else:
                    acc_accepted = 0.0
                    
                # Decomposição da abstenção: cobertura sobre ID (quanto do serviço
                # é mantido) e rejeição sobre OOD (proteção emergente).
                id_coverage = (~abstained[id_mask]).float().mean().item() if id_mask.sum() > 0 else 0.0
                ood_rejection = abstained[ood_mask].float().mean().item() if ood_mask.sum() > 0 else 0.0

                prefix = "" if risk_pct == 10 else f"_{risk_pct}"
                sgr_extras.update({
                    f"sgr_applied_threshold{prefix}": sgr_th,
                    f"sgr_abstention_rate{prefix}": abstention_rate,
                    f"sgr_accepted_accuracy{prefix}": acc_accepted,
                    f"sgr_id_coverage{prefix}": id_coverage,
                    f"sgr_ood_rejection_rate{prefix}": ood_rejection
                })
        
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
    def evaluate_ood_margin(self, dataset, support_text: list, labels_dict: dict,
                             val_dataset=None, batch_size=32, save_dir='./results'):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]

        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = [item['class_id'] for item in batch]
            return texts, torch.tensor(labels, dtype=torch.long)

        model = self.models[0]
        kshot = self.config.get('sampler', {}).get('kshot', 5)

        def _score_split(dl):
            """Calcula o score MARGIN (2ª menor dist - menor dist) para um split."""
            split_preds, split_confidences, split_targets = [], [], []
            with torch.no_grad():
                for texts, labels in tqdm(dl):
                    input_text = support_text + list(texts)
                    model_outputs = model(input_text, label_text_list, kshot=kshot)
                    prototypes = model_outputs[0]
                    query_embeddings = model_outputs[1]

                    # Mesmo score canônico do DEFAULT (methods/laqda/scoring.py)
                    dists, _, _, _ = compute_prototype_scores(query_embeddings, prototypes)

                    # Ordena distâncias em ordem crescente (1º = mais próximo)
                    sorted_dists, sorted_idx = torch.sort(dists, dim=1)
                    min_dists = sorted_dists[:, 0]
                    preds = sorted_idx[:, 0]

                    if sorted_dists.shape[1] >= 2:
                        # Margem: quanto maior, mais "isolada" está a predição no espaço de embeddings
                        second_dists = sorted_dists[:, 1]
                        confidences = second_dists - min_dists
                    else:
                        # Fallback para caso binário com apenas 1 classe (edge case)
                        confidences = -min_dists

                    split_preds.append(preds.cpu())
                    split_confidences.append(confidences.cpu())
                    split_targets.append(labels.cpu())
            return torch.cat(split_preds), torch.cat(split_confidences), torch.cat(split_targets)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print("Avaliando OOD metrics — Estratégia: Distance Margin (test_final_margin)...")
        preds, confidences, all_targets_t = _score_split(dataloader)

        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]

        # SGR próprio do MARGIN: a margem (2ª menor dist - menor dist) está numa escala
        # diferente do score DEFAULT (-min_dist), então os thresholds do modelo (buffers)
        # não transferem — calibramos aqui no val set (só ID), usando este mesmo score.
        sgr_extras = {}
        if val_dataset is not None:
            print("Distance Margin: calibrando thresholds SGR no validation set...")
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            val_preds, val_confidences, val_targets_t = _score_split(val_loader)
            val_id_mask = val_targets_t != -1

            sgr_extras.update(self._fit_and_apply_sgr(
                val_confidences[val_id_mask], val_preds[val_id_mask], val_targets_t[val_id_mask],
                confidences, preds, all_targets_t, id_mask, ood_mask
            ))
        else:
            print("  [Aviso] val_dataset não fornecido — SGR não aplicado ao Distance Margin.")

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
            **sgr_extras
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
                                 val_dataset=None, batch_size=32, save_dir='./results', n_passes: int = 20):
        id2label = {v: k for k, v in labels_dict.items()}
        label_text_list = [id2label[i] for i in range(len(labels_dict))]

        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            labels = [item['class_id'] for item in batch]
            return texts, torch.tensor(labels, dtype=torch.long)

        model = self.models[0]
        kshot = self.config.get('sampler', {}).get('kshot', 5)

        def _score_split(dl):
            """Calcula o score MC Dropout (mean_sim - var_sim, N passes) para um split."""
            split_preds, split_confidences, split_targets = [], [], []
            # Ativar dropout estocástico SEM ligar o modo treino do LaqdaModule: model.train()
            # (chamado no nn.Module raiz) cascateia self.training=True pro laqda_module também,
            # o que reativa o TransductiveQDASampler (só deveria rodar durante o loop de
            # treino real — ver comentário em laqda_module.py). Em inferência o batch de
            # query não tem o formato de episódio (nway*qshot contíguo) que o sampler espera,
            # e ele quebra com RuntimeError de reshape/topk assim que o batch não bate esse
            # formato. model.eval() primeiro mantém LaqdaModule.training=False (sampler
            # desligado); religamos só as camadas de Dropout depois, que é o que realmente
            # precisa ficar estocástico para o MC Dropout funcionar.
            model.eval()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()
            with torch.no_grad():
                for texts, labels in tqdm(dl):
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

                    split_preds.append(preds_idx)
                    split_confidences.append(confidence)
                    split_targets.append(labels.cpu())
            # Voltar para modo eval após os passes MC
            model.eval()
            return torch.cat(split_preds), torch.cat(split_confidences), torch.cat(split_targets)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"Avaliando OOD metrics — Estratégia: MC Dropout ({n_passes} passes) (test_final_mcdropout)...")
        all_preds_t, confidences, all_targets_t = _score_split(dataloader)

        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]

        # SGR próprio do MC Dropout: mean_sim - var_sim está numa escala diferente do
        # score DEFAULT (-min_dist x tau), então os thresholds do modelo (buffers) não
        # transferem — calibramos aqui no val set (só ID), usando este mesmo score.
        sgr_extras = {}
        if val_dataset is not None:
            print("MC Dropout: calibrando thresholds SGR no validation set...")
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            val_preds, val_confidences, val_targets_t = _score_split(val_loader)
            val_id_mask = val_targets_t != -1

            sgr_extras.update(self._fit_and_apply_sgr(
                val_confidences[val_id_mask], val_preds[val_id_mask], val_targets_t[val_id_mask],
                confidences, all_preds_t, all_targets_t, id_mask, ood_mask
            ))
        else:
            print("  [Aviso] val_dataset não fornecido — SGR não aplicado ao MC Dropout.")

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
            **sgr_extras
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
        have_val_scores = False  # sinaliza se val_logits_t/val_targets_t ficaram disponíveis

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
                    have_val_scores = True
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

        # SGR próprio do Temperature Scaling: a confiança é uma probabilidade softmax
        # calibrada por tau_opt, escala diferente do score DEFAULT (-min_dist x 15.0)
        # — os thresholds do modelo (buffers) não transferem. Reaproveita o mesmo
        # forward pass do val já feito acima para calibrar tau_opt (val_logits_t /
        # val_targets_t), sem rodar o encoder no val set de novo.
        sgr_extras = {}
        if have_val_scores:
            val_probs = F.softmax(val_logits_t * tau_opt, dim=1)
            val_confidences, val_preds = torch.max(val_probs, dim=1)
            val_id_mask = val_targets_t != -1

            sgr_extras.update(self._fit_and_apply_sgr(
                val_confidences[val_id_mask], val_preds[val_id_mask], val_targets_t[val_id_mask],
                confidences, preds, all_targets_t, id_mask, ood_mask
            ))
        else:
            print("  [Aviso] val_dataset não fornecido/vazio — SGR não aplicado ao Temperature Scaling.")

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
            **sgr_extras
        )
        print(f"Relatório test_final_tempscale_metrics_report.json salvo em {save_dir}")

    # =========================================================================
    # ESTRATÉGIA 4: X-Mahalanobis Feature Mixing
    # Usa o Support Set para estimar as matrizes de covariância (pesos) e
    # Mahalanobis global, adaptando o artigo para Few-Shot/LAQDA.
    # =========================================================================
    def evaluate_ood_xmahalanobis(self, dataset, support_text: list, labels_dict: dict,
                                   batch_size=32, save_dir='./results', val_dataset=None):
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
                _, all_hidden = model.encoder(texts, label_text_list, output_hidden_states=True)
                stacked_hidden = torch.stack(all_hidden, dim=0).cpu()
                support_all_hidden.append(stacked_hidden)
                support_labels.append(labels)
                
        support_all_hidden = torch.cat(support_all_hidden, dim=1)
        support_labels = torch.cat(support_labels)
        
        num_layers, total_support, hidden_size = support_all_hidden.shape

        # Normaliza cada camada para escala comparável (L2 por amostra) ANTES de medir
        # variância e de misturar. Sem isso, uma camada com ativações de magnitude bruta
        # maior (aqui, a camada Label-Aware customizada — encoder.py, sem LayerNorm final)
        # domina o peso só por escala, não por conteúdo discriminativo, e o mecanismo do
        # paper (aproveitar sinal das camadas rasas) fica anulado na prática.
        support_all_hidden = F.normalize(support_all_hidden, p=2, dim=2)

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

        # Covariância shrinkage (Ledoit-Wolf) em vez de MLE + jitter fixo: com
        # total_support (kshot * nway, tipicamente 30-60) << hidden_size (768), a
        # covariância MLE é mal-condicionada (posto <= total_support - n_classes,
        # deixando centenas de direções com variância ~0) e um jitter de 1e-6 é
        # regularização irrisória — a inversa amplifica ruído numérico por ~1e6 nessas
        # direções. Ledoit-Wolf estima o shrinkage ótimo para esse regime
        # alta-dimensão/poucas-amostras.
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=True).fit(centered_mixed.numpy())

        # Piso de regularização absoluto: com kshot=1, cada classe tem 1 única amostra,
        # então centered_mixed é EXATAMENTE zero (amostra - média da própria classe) —
        # não há sinal de variância intra-classe algum. Nesse caso o próprio Ledoit-Wolf
        # escolhe shrinkage=0 (não há nada para encolher em direção à identidade: seu
        # alvo também é derivado de centered_mixed, logo também é zero) e a covariância
        # shrinkada colapsa para 0, zerando a inversa inteira -> toda distância de
        # Mahalanobis vira 0 para todas as classes (AUROC=0.5, tudo empatado). O piso
        # usa a escala da variância GERAL entre amostras (support_mixed), que nunca é
        # zero mesmo com 1 amostra por classe, em vez de depender do sinal intra-classe.
        floor = 1e-3 * support_mixed.var(dim=0).mean().item()
        cov = torch.tensor(lw.covariance_, dtype=torch.float32) + floor * torch.eye(hidden_size)
        inv_cov = torch.linalg.inv(cov)

        device = next(model.parameters()).device
        alphas_t = alphas_t.to(device).squeeze(2) # (num_layers, 1)
        class_means = class_means.to(device)
        inv_cov = inv_cov.to(device)

        def _score_split(dl):
            """Aplica o score X-Mahalanobis (parâmetros fixados no support) a um split."""
            split_dists, split_targets = [], []
            with torch.no_grad():
                for texts, labels in tqdm(dl):
                    _, all_hidden = model.encoder(texts, label_text_list, output_hidden_states=True)
                    stacked_hidden = torch.stack(all_hidden, dim=0)
                    stacked_hidden = F.normalize(stacked_hidden, p=2, dim=2)

                    query_mixed = (stacked_hidden * alphas_t.unsqueeze(2)).sum(dim=0)

                    batch_size_cur = query_mixed.shape[0]
                    dists = torch.zeros(batch_size_cur, n_classes, device=device)
                    for c in range(n_classes):
                        diff = query_mixed - class_means[c].unsqueeze(0)
                        left = torch.matmul(diff, inv_cov)
                        mahalanobis_dist = (left * diff).sum(dim=1)
                        dists[:, c] = mahalanobis_dist

                    split_dists.append(dists.cpu())
                    split_targets.append(labels.cpu())
            return torch.cat(split_dists), torch.cat(split_targets)

        print(f"Avaliando OOD metrics — Estratégia: X-Mahalanobis (test_final_xmaha)...")
        all_dists_t, all_targets_t = _score_split(dataloader)

        min_dists, preds = torch.min(all_dists_t, dim=1)
        confidences = -min_dists

        id_mask = all_targets_t != -1
        ood_mask = all_targets_t == -1
        id_scores = confidences[id_mask]
        ood_scores = confidences[ood_mask]

        # id_only_accuracy é calculado centralmente em compute_accuracy_f1
        # (methods/metrics/classification.py) a partir de targets/preds.
        sgr_extras = {}

        # Calibração SGR held-out para o score DEFAULT. Antes, este caminho lia
        # thresholds dos buffers do modelo, que são gravados no fim do treino
        # sobre as queries episódicas de validação -- o mesmo fluxo monitorado
        # pelo early stopping, portanto NÃO disjunto. Aqui calibramos no split
        # de validação (só ID) e aplicamos ao teste sem alteração, que é a
        # condição que a garantia exige.
        if val_dataset is not None:
            print("Default score: calibrando thresholds SGR no validation set (held-out)...")
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            v_dists, v_targets = [], []
            with torch.no_grad():
                for texts, labels in tqdm(val_loader):
                    input_text = support_text + list(texts)
                    kshot = self.config.get('sampler', {}).get('kshot', 5)
                    mo = model(input_text, label_text_list, kshot=kshot)
                    d, _, _, _ = compute_prototype_scores(mo[1], mo[0])
                    v_dists.append(d.cpu()); v_targets.append(labels.cpu())
            v_dists_t = torch.cat(v_dists); v_targets_t = torch.cat(v_targets)
            v_min, v_preds = torch.min(v_dists_t, dim=1)
            v_conf = -v_min
            v_id = v_targets_t != -1
            sgr_extras.update(self._fit_and_apply_sgr(
                v_conf[v_id], v_preds[v_id], v_targets_t[v_id],
                confidences, preds, all_targets_t, id_mask, ood_mask
            ))

        # SGR próprio do X-Mahalanobis: os thresholds salvos no modelo (buffers)
        # foram calibrados no score canônico (cosseno x tau) e NÃO transferem
        # para a escala Mahalanobis. Calibramos aqui thresholds específicos no
        # val set (somente ID), usando EXATAMENTE este score, e aplicamos ao teste.
        if val_dataset is not None:
            print("X-Mahalanobis: calibrando thresholds SGR no validation set...")
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            val_dists_t, val_targets_t = _score_split(val_loader)

            val_id_mask = val_targets_t != -1
            val_min_dists, val_preds = torch.min(val_dists_t, dim=1)
            val_confidences = -val_min_dists

            sgr_extras.update(self._fit_and_apply_sgr(
                val_confidences[val_id_mask], val_preds[val_id_mask], val_targets_t[val_id_mask],
                confidences, preds, all_targets_t, id_mask, ood_mask
            ))
        else:
            print("  [Aviso] val_dataset não fornecido — SGR não aplicado ao X-Mahalanobis "
                  "(os thresholds do modelo estão em outra escala e não podem ser reutilizados).")

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
