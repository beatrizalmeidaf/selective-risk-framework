import os
import torch
import json
import zipfile
from tqdm import tqdm
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
                    
                    dists = torch.pow(query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0), 2).sum(2)
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
                
                dists = torch.pow(query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0), 2).sum(2)
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
        
        from methods.metrics.reporter import MetricsReporter
        reporter = MetricsReporter(save_dir=save_dir)
        reporter.generate_report(
            confidences=confidences,
            preds=preds,
            targets=all_targets_t,
            id_scores=id_scores,
            ood_scores=ood_scores,
            model=model,
            prefix="test_final"
        )
        print(f"Relatório test_final_metrics_report.json salvo em {save_dir}")
