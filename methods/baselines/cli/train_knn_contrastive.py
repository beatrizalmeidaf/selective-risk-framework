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

from methods.baselines.distance.knn.knn_contrastive_model import KNNContrastiveModel
from methods.baselines.distance.knn.knn_contrastive_loss import KNNContrastiveLoss
from methods.baselines.distance.knn.lof_scorer import LOFScorer
from methods.baselines.distance.knn.scorer import KNNScorer
from methods.baselines.sota.conjnorm.scorer import ConjNormScorer

def get_parser():
    parser = argparse.ArgumentParser(description="Treinamento de KNN-Contrastive Learning para Avaliação OOD (Zhou et al.)")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Caminho base do dataset')
    parser.add_argument('--fold', type=str, required=True, help='Identificador do fold (ex: 01)')
    parser.add_argument('--save_dir', type=str, default='./outputs/knn_contrastive', help='Diretório para salvar modelo')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do lote (Batch Size)')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de treinamento')
    parser.add_argument('--lr', type=float, default=2e-5, help='Taxa de aprendizado')
    parser.add_argument('--kshot', type=int, default=None, help='Número de shots por classe')
    parser.add_argument('--patience', type=int, default=20, help='Patience para Early Stopping')
    
    # KNN-Contrastive Args
    parser.add_argument('--k_neighbors', type=int, default=5, help='K-vizinhos da mesma classe (para KNN Loss)')
    parser.add_argument('--queue_size', type=int, default=256, help='Tamanho da fila do MoCo (reduzido para few-shot)')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momentum para atualização do key encoder')
    parser.add_argument('--tau', type=float, default=0.07, help='Temperatura para similaridade de cosseno')
    parser.add_argument('--lam', type=float, default=0.5, help='Peso Lambda para combinar L_knn-cl e L_ce')
    return parser

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = [item['class_id'] for item in batch]
    return texts, torch.tensor(labels, dtype=torch.long)

def main():
    args = get_parser().parse_args()
    
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    
    global_config_path = 'configs/model_encoder_config.yaml'
    if os.path.exists(global_config_path):
        global_config = load_config(global_config_path)
        lang = global_config.get('model', {}).get('active_language', 'pt')
        global_model_name = global_config.get('model', {}).get(f'encoder_name_{lang}', 'bert-base-uncased')
    else:
        global_model_name = 'bert-base-uncased'
        
    print(f"Instanciando KNN-Contrastive Model com Encoder Universal: {global_model_name}")

    methods_config_path = 'configs/methods_config.yaml'
    if args.kshot is None and os.path.exists(methods_config_path):
        methods_config = load_config(methods_config_path)
        args.kshot = methods_config.get('baselines', {}).get('kshots', [5])[0]

    if args.kshot is not None:
        args.save_dir = os.path.join(args.save_dir, f'kshot_{args.kshot}')

    os.makedirs(args.save_dir, exist_ok=True)

    datamodule = StandardDataModule(args.dataset_dir, args.fold, batch_size=args.batch_size, keep_unknown_classes=True, kshot=args.kshot)
    datamodule.setup()
    labels_dict = datamodule.labels_dict
    
    train_loader = datamodule.train_dataloader(collate_fn=collate_fn)
    valid_loader = datamodule.val_dataloader(collate_fn=collate_fn)
    
    # Ajustar queue_size para o tamanho do dataset se for menor
    total_train_samples = len(train_loader.dataset)
    if args.queue_size > total_train_samples:
        # A queue size deve ser divisível pelo batch size
        adjusted_queue = max(16, (total_train_samples // args.batch_size) * args.batch_size)
        print(f"Ajustando queue_size de {args.queue_size} para {adjusted_queue} para caber no few-shot sem padding excessivo")
        args.queue_size = adjusted_queue

    num_classes = len(labels_dict)
    model = KNNContrastiveModel(
        global_model_name, num_classes, 
        dim=768, queue_size=args.queue_size, momentum=args.momentum
    )
    model.to(device_str)
    
    optimizer = AdamW(model.encoder_q.parameters(), lr=args.lr)
    
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    knn_loss_fn = KNNContrastiveLoss(k_neighbors=args.k_neighbors, tau=args.tau)
    
    best_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_ce, total_knn = 0.0, 0.0, 0.0
        correct, total = 0, 0
        
        for texts, labels in tqdm(train_loader, desc=f"Train Ep {epoch}"):
            labels = labels.to(device_str)
            optimizer.zero_grad()
            
            # Forward pass: obtém query, key e queue
            features_q, logits_q, features_k, queue, queue_labels = model(texts, labels)
            
            # Calcula as duas losses
            l_ce = ce_loss_fn(logits_q, labels)
            l_knn = knn_loss_fn(features_q, features_k, queue, queue_labels, labels)
            
            # Objetivo combinado
            loss = args.lam * l_knn + (1 - args.lam) * l_ce
            
            loss.backward()
            optimizer.step()
            
            # Enqueue e Dequeue (Atualiza a fila)
            model._dequeue_and_enqueue(features_k, labels)
            
            total_loss += loss.item()
            total_ce += l_ce.item()
            total_knn += l_knn.item()
            
            preds = torch.argmax(logits_q, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total
        print(f"Train - Loss: {total_loss/len(train_loader):.4f} (CE: {total_ce/len(train_loader):.4f} | KNN: {total_knn/len(train_loader):.4f}), Acc={train_acc:.4f}")
        
        # Validação
        if valid_loader:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for texts, labels in tqdm(valid_loader, desc=f"Val Ep {epoch}"):
                    labels = labels.to(device_str)
                    
                    features_q, logits_q = model(texts)
                    loss = ce_loss_fn(logits_q, labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits_q, dim=-1)
                    
                    mask_id = (labels != -1)
                    val_correct += (preds[mask_id] == labels[mask_id]).sum().item()
                    val_total += mask_id.sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            print(f"Média Val: Loss CE={val_loss/len(valid_loader):.4f}, Acc={val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_knn_contrastive.pth'))
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}. No improvement for {args.patience} epochs.")
                break

    # ==========================================
    # Avaliação Final no Conjunto de Teste
    # ==========================================
    print("Iniciando avaliação final no conjunto de Teste...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_knn_contrastive.pth')))
    model.eval()
    
    test_loader = datamodule.test_dataloader(collate_fn=collate_fn)
    if test_loader:
        all_features = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in tqdm(test_loader, desc="Test Inference"):
                labels = labels.to(device_str)
                features_q, logits_q = model(texts)
                
                all_features.append(features_q.cpu())
                all_logits.append(logits_q.cpu())
                all_labels.append(labels.cpu())
                
        test_features_t = torch.cat(all_features)
        test_logits_t = torch.cat(all_logits)
        test_labels_t = torch.cat(all_labels)

        # Extrair features do conjunto de treino para LOF/KNN
        print("Extraindo features de treino para OOD Scorers...")
        train_features_list = []
        train_labels_list = []
        with torch.no_grad():
            for texts, labels in tqdm(train_loader, desc="Train Features"):
                labels = labels.to(device_str)
                features_q, _ = model(texts)
                train_features_list.append(features_q.cpu())
                train_labels_list.append(labels.cpu())
        
        train_features_t = torch.cat(train_features_list)
        train_labels_t = torch.cat(train_labels_list)
        
        # Em cenário few-shot, n_neighbors e k DEVEM ser limitados pelo número de amostras da classe
        eval_k = args.kshot if (args.kshot is not None and args.kshot > 0) else 10
        # Garante no mínimo 3 vizinhos para o LOF funcionar estatisticamente (ou o total da classe)
        eval_k = max(3, eval_k)

        print(f"Usando k={eval_k} para Scorers OOD baseado no kshot={args.kshot}")
        
        # Ajustar Scorers
        lof_cosine = LOFScorer(n_neighbors=eval_k, metric='cosine')
        lof_cosine.fit(train_features_t, train_labels_t)
        
        lof_euclidean = LOFScorer(n_neighbors=eval_k, metric='euclidean')
        lof_euclidean.fit(train_features_t, train_labels_t)
        
        knn_scorer = KNNScorer(k=eval_k, metric='cosine')
        knn_scorer.fit(train_features_t, train_labels_t)
        
        conjnorm_scorer = ConjNormScorer(p=2.5, alpha=1.0)
        conjnorm_scorer.fit(train_features_t, train_labels_t)
        
        reporter = MetricsReporter(save_dir=args.save_dir)
        
        # Predições de ID baseadas no CrossEntropy (Baseline)
        probs = F.softmax(test_logits_t, dim=-1)
        _, preds = torch.max(probs, dim=-1)
        
        mask_id = (test_labels_t != -1)
        mask_ood = (test_labels_t == -1)
        
        # 1. Avaliar LOF Cosine
        lof_cos_conf = lof_cosine.compute_score(test_features_t)
        reporter.generate_report(
            confidences=lof_cos_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=lof_cos_conf[mask_id],
            ood_scores=lof_cos_conf[mask_ood],
            model=model,
            prefix="test_final_lof_cosine"
        )

        # 2. Avaliar LOF Euclidean
        lof_euc_conf = lof_euclidean.compute_score(test_features_t)
        reporter.generate_report(
            confidences=lof_euc_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=lof_euc_conf[mask_id],
            ood_scores=lof_euc_conf[mask_ood],
            model=model,
            prefix="test_final_lof_euclidean"
        )
        
        # 3. Avaliar KNN Contrastivo Clássico
        knn_conf = knn_scorer.compute_score(test_features_t)
        reporter.generate_report(
            confidences=knn_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=knn_conf[mask_id],
            ood_scores=knn_conf[mask_ood],
            model=model,
            prefix="test_final_knn_contrastive_scorer"
        )
        
        # 4. Avaliar ConjNorm
        conjnorm_conf = conjnorm_scorer.compute_score(test_features_t)
        reporter.generate_report(
            confidences=conjnorm_conf,
            preds=preds,
            targets=test_labels_t,
            id_scores=conjnorm_conf[mask_id],
            ood_scores=conjnorm_conf[mask_ood],
            model=model,
            prefix="test_final_conjnorm"
        )

        print("Avaliação KNN-Contrastive com todos os Scorers concluída!")

if __name__ == "__main__":
    main()
