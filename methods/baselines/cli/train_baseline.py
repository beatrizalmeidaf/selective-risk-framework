import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.datasets.jsonl_dataset import JSONLDataset
from methods.laqda.utils.config_loader import load_config
from methods.baselines.models.standard_classifier import BaselineClassifier

def get_parser():
    parser = argparse.ArgumentParser(description="Treinamento de Baseline Genérico para Avaliação OOD")
    parser.add_argument('--train_file', type=str, required=True, help='Caminho para arquivo JSONL de treino')
    parser.add_argument('--valid_file', type=str, help='Caminho para arquivo JSONL de validação')
    parser.add_argument('--save_dir', type=str, default='./outputs/baseline', help='Diretório para salvar modelo e tensores')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do lote (Batch Size)')
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de treinamento')
    parser.add_argument('--lr', type=float, default=2e-5, help='Taxa de aprendizado')
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

    # 2. Datasets
    temp_dataset = JSONLDataset(args.train_file)
    labels_dict = {str(c): i for i, c in enumerate(temp_dataset.get_classes())}
    train_dataset = JSONLDataset(args.train_file, class_name_to_id=labels_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    valid_loader = None
    if args.valid_file and os.path.exists(args.valid_file):
        valid_dataset = JSONLDataset(args.valid_file, class_name_to_id=labels_dict)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Modelo e Otimizador
    num_classes = len(labels_dict)
    model = BaselineClassifier(global_model_name, num_classes)
    model.to(device_str)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
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
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            print(f"Média Val: Loss={val_loss/len(valid_loader):.4f}, Acc={val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_baseline.pth'))
                
                # Salvar tensores para avaliação posterior dos OOD Scorers
                torch.save(torch.cat(all_features), os.path.join(args.save_dir, 'val_features.pt'))
                torch.save(torch.cat(all_logits), os.path.join(args.save_dir, 'val_logits.pt'))
                torch.save(torch.cat(all_labels), os.path.join(args.save_dir, 'val_labels.pt'))

            # Padrão Ouro de Avaliação
            from methods.metrics.reporter import MetricsReporter
            reporter = MetricsReporter(save_dir=args.save_dir)
            
            all_logits_t = torch.cat(all_logits)
            all_labels_t = torch.cat(all_labels)
            
            # Para o baseline, a confiança é o max do softmax
            import torch.nn.functional as F
            probs = F.softmax(all_logits_t, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            
            reporter.generate_report(
                confidences=confidences,
                preds=preds,
                targets=all_labels_t,
                model=model,
                prefix=f"val_ep_{epoch}"
            )

if __name__ == "__main__":
    main()
