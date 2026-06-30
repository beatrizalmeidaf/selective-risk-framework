import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.datamodule import StandardDataModule
from methods.laqda.utils.config_loader import load_config
from methods.metrics.reporter import MetricsReporter
from methods.baselines.models.standard_classifier import BaselineClassifier
from methods.baselines.energy_score.scorer import EnergyScorer
from methods.baselines.distance.mahalanobis.scorer import MahalanobisScorer
from methods.baselines.distance.knn.scorer import KNNScorer
from methods.baselines.sota.gradnorm.scorer import GradNormScorer
from methods.baselines.sota.pruning.react_scorer import ReActScorer
from methods.baselines.sota.conjnorm.scorer import ConjNormScorer

def get_parser():
    parser = argparse.ArgumentParser(description="Treinamento de Baseline Genérico para Avaliação OOD")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Caminho base do dataset')
    parser.add_argument('--fold', type=str, required=True, help='Identificador do fold (ex: 01)')
    parser.add_argument('--save_dir', type=str, default='./outputs/baseline', help='Diretório para salvar modelo e tensores')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do lote (Batch Size)')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de treinamento')
    parser.add_argument('--lr', type=float, default=2e-5, help='Taxa de aprendizado')
    parser.add_argument('--kshot', type=int, default=None, help='Número de shots por classe (opcional, sobrescreve config)')
    return parser

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = [item['class_id'] for item in batch]
    return texts, torch.tensor(labels, dtype=torch.long)

def main():
    args = get_parser().parse_args()
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    
    # 1. Carregar configuração global de arquitetura
    global_config_path = 'configs/model_encoder_config.yaml'
    if os.path.exists(global_config_path):
        global_config = load_config(global_config_path)
        lang = global_config.get('model', {}).get('active_language', 'pt')
        global_model_name = global_config.get('model', {}).get(f'encoder_name_{lang}', 'bert-base-uncased')
    else:
        global_model_name = 'bert-base-uncased'
        
    print(f"Instanciando Baseline com Encoder Universal: {global_model_name}")

    # Extrair kshot do config caso não venha do argumento
    methods_config_path = 'configs/methods_config.yaml'
    if args.kshot is None and os.path.exists(methods_config_path):
        methods_config = load_config(methods_config_path)
        args.kshot = methods_config.get('baselines', {}).get('kshots', [5])[0]

    if args.kshot is not None:
        args.save_dir = os.path.join(args.save_dir, f'kshot_{args.kshot}')

    # 2. Datasets
    datamodule = StandardDataModule(args.dataset_dir, args.fold, batch_size=args.batch_size, keep_unknown_classes=True, kshot=args.kshot)
    datamodule.setup()
    labels_dict = datamodule.labels_dict
    
    train_loader = datamodule.train_dataloader(collate_fn=collate_fn)
    valid_loader = datamodule.val_dataloader(collate_fn=collate_fn)

    # 3. Modelo e Otimizador
    num_classes = len(labels_dict)
    model = BaselineClassifier(global_model_name, num_classes)
    model.to(device_str)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. Loop de Treinamento
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for texts, labels in tqdm(train_loader, desc=f"Train Ep {epoch}"):
            labels = labels.to(device_str)
            optimizer.zero_grad()
            
            features, logits = model(texts)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total
        print(f"Média Train: Loss={total_loss/len(train_loader):.4f}, Acc={train_acc:.4f}")
        
        # Validação
        if valid_loader:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            all_features = []
            all_logits = []
            all_labels = []
            
            with torch.no_grad():
                for texts, labels in tqdm(valid_loader, desc=f"Val Ep {epoch}"):
                    labels = labels.to(device_str)
                    features, logits = model(texts)
                    
                    all_features.append(features.cpu())
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())
                    
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=-1)
                    
                    # Ignorar OOD (-1) no calculo de acuracia de validação (ID)
                    mask_id = (labels != -1)
                    val_correct += (preds[mask_id] == labels[mask_id]).sum().item()
                    val_total += mask_id.sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            print(f"Média Val: Loss={val_loss/len(valid_loader):.4f}, Acc={val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_baseline.pth'))
                
                # Salvar tensores para avaliação posterior dos OOD Scorers
                torch.save(torch.cat(all_features), os.path.join(args.save_dir, 'val_features.pt'))
                torch.save(torch.cat(all_logits), os.path.join(args.save_dir, 'val_logits.pt'))
                torch.save(torch.cat(all_labels), os.path.join(args.save_dir, 'val_labels.pt'))

            # Padrão Ouro de Avaliação
            reporter = MetricsReporter(save_dir=args.save_dir)
            
            all_logits_t = torch.cat(all_logits)
            all_labels_t = torch.cat(all_labels)
            
            # Para o baseline, a confiança (certeza ID) é o valor máximo do softmax
            probs = F.softmax(all_logits_t, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            
            val_mask_id = (all_labels_t != -1)
            val_mask_ood = (all_labels_t == -1)
            
            reporter.generate_report(
                confidences=confidences,
                preds=preds,
                targets=all_labels_t,
                id_scores=confidences[val_mask_id],
                ood_scores=confidences[val_mask_ood],
                model=model,
                prefix=f"val_ep_{epoch}"
            )

    # ==========================================
    # 5. Avaliação Final no Conjunto de Teste
    # ==========================================
    print("Iniciando avaliação final no conjunto de Teste...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_baseline.pth')))
    model.eval()
    
    test_loader = datamodule.test_dataloader(collate_fn=collate_fn)
    if test_loader:
        all_features = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in tqdm(test_loader, desc="Test Inference"):
                labels = labels.to(device_str)
                features, logits = model(texts)
                
                all_features.append(features.cpu())
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                
        # Salvar tensores de Teste para avaliação dos OOD Scorers
        torch.save(torch.cat(all_features), os.path.join(args.save_dir, 'test_features.pt'))
        torch.save(torch.cat(all_logits), os.path.join(args.save_dir, 'test_logits.pt'))
        torch.save(torch.cat(all_labels), os.path.join(args.save_dir, 'test_labels.pt'))

        # Extrair features do conjunto de treino (necessário para Mahalanobis e KNN)
        print("Extraindo features de treino para Mahalanobis e KNN...")
        train_features_list = []
        train_labels_list = []
        with torch.no_grad():
            for texts, labels in tqdm(train_loader, desc="Train Features for OOD"):
                labels = labels.to(device_str)
                features, _ = model(texts)
                train_features_list.append(features.cpu())
                train_labels_list.append(labels.cpu())
        
        train_features_t = torch.cat(train_features_list)
        train_labels_t = torch.cat(train_labels_list)
        
        # Ajustar Scorers de Distância
        mahalanobis = MahalanobisScorer()
        mahalanobis.fit(train_features_t, train_labels_t)
        
        methods_config_path = 'configs/methods_config.yaml'
        if os.path.exists(methods_config_path):
            methods_config = load_config(methods_config_path)
            baselines_config = methods_config.get('baselines', {})
            knn_k = baselines_config.get('knn', {}).get('k', 50)
            energy_temp = baselines_config.get('energy_score', {}).get('temperature', 1.0)
        else:
            knn_k = 50
            energy_temp = 1.0
            
        knn = KNNScorer(k=knn_k, metric='euclidean')
        knn.fit(train_features_t, train_labels_t)
        
        knn_cont = KNNScorer(k=knn_k, metric='cosine')
        knn_cont.fit(train_features_t, train_labels_t)
        
        energy_scorer = EnergyScorer(temperature=energy_temp)
        
        # Ajustar Scorers SOTA
        gradnorm_scorer = GradNormScorer(num_classes)
        react_scorer = ReActScorer(model.classifier)
        react_scorer.fit(train_features_t)
        conjnorm_scorer = ConjNormScorer(model.classifier)
        
        # Preparar dados de Teste
        reporter = MetricsReporter(save_dir=args.save_dir)
        test_features_t = torch.cat(all_features)
        test_logits_t = torch.cat(all_logits)
        test_labels_t = torch.cat(all_labels)
        
        # Probabilidades Softmax usadas pelo método Baseline (MSP)
        probs = F.softmax(test_logits_t, dim=-1)
        _, preds = torch.max(probs, dim=-1)
        
        mask_id = (test_labels_t != -1)
        mask_ood = (test_labels_t == -1)
        
        # 1. Avaliar MSP (Maximum Softmax Probability)
        msp_confidences, _ = torch.max(probs, dim=-1)
        reporter.generate_report(
            confidences=msp_confidences,
            preds=preds,
            targets=test_labels_t,
            id_scores=msp_confidences[mask_id],
            ood_scores=msp_confidences[mask_ood],
            model=model,
            prefix="test_final_msp"
        )
        
        # 2. Avaliar Energy Score
        energy_conf = energy_scorer.compute_score(test_logits_t)
        reporter.generate_report(
            confidences=energy_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=energy_conf[mask_id],
            ood_scores=energy_conf[mask_ood],
            model=model,
            prefix="test_final_energy"
        )
        
        # 3. Avaliar Mahalanobis
        maha_conf = mahalanobis.compute_score(test_features_t)
        reporter.generate_report(
            confidences=maha_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=maha_conf[mask_id],
            ood_scores=maha_conf[mask_ood],
            model=model,
            prefix="test_final_mahalanobis"
        )
        
        # 4. Avaliar KNN Normal
        knn_conf = knn.compute_score(test_features_t)
        reporter.generate_report(
            confidences=knn_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=knn_conf[mask_id],
            ood_scores=knn_conf[mask_ood],
            model=model,
            prefix="test_final_knn"
        )
        
        # 5. Avaliar KNN Contrastivo
        knn_cont_conf = knn_cont.compute_score(test_features_t)
        reporter.generate_report(
            confidences=knn_cont_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=knn_cont_conf[mask_id],
            ood_scores=knn_cont_conf[mask_ood],
            model=model,
            prefix="test_final_knn_contrastive"
        )
        
        # 6. Avaliar GradNorm
        gradnorm_conf = gradnorm_scorer.compute_score(test_features_t, test_logits_t)
        reporter.generate_report(
            confidences=gradnorm_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=gradnorm_conf[mask_id],
            ood_scores=gradnorm_conf[mask_ood],
            model=model,
            prefix="test_final_sota_gradnorm"
        )
        
        # 7. Avaliar ReAct
        react_conf = react_scorer.compute_score(test_features_t)
        reporter.generate_report(
            confidences=react_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=react_conf[mask_id],
            ood_scores=react_conf[mask_ood],
            model=model,
            prefix="test_final_sota_react"
        )
        
        # 8. Avaliar ConjNorm
        conjnorm_conf = conjnorm_scorer.compute_score(test_features_t)
        reporter.generate_report(
            confidences=conjnorm_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=conjnorm_conf[mask_id],
            ood_scores=conjnorm_conf[mask_ood],
            model=model,
            prefix="test_final_sota_conjnorm"
        )
        
        print("Avaliação de Teste com todos os Scorers OOD concluída!")

if __name__ == "__main__":
    main()
