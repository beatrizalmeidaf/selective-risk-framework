import argparse
import os
import torch
from data.datasets.jsonl_dataset import JSONLDataset
from ..utils.config_loader import load_config
from ..inference.infer import LaqdaInferencer

def get_parser():
    parser = argparse.ArgumentParser(description="Inferência LAQDA")
    parser.add_argument('--config', type=str, default='methods/laqda/configs/default.yaml')
    parser.add_argument('--test_file', type=str, required=True, help='Arquivo JSONL de teste')
    parser.add_argument('--train_file', type=str, required=True, help='Arquivo JSONL de treino (âncoras)')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Caminho(s) para os pesos')
    parser.add_argument('--output_dir', type=str, default='./results', help='Diretório de saída para predições brutas')
    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config)
    
    device = config.get('hardware', {}).get('device', 0)
    device_str = f'cuda:{device}' if torch.cuda.is_available() and device >= 0 else 'cpu'
    
    # Prepara datasets
    train_dataset = JSONLDataset(args.train_file)
    labels_dict = {str(c): i for i, c in enumerate(train_dataset.get_classes())}
    
    test_dataset = JSONLDataset(args.test_file)
    
    inferencer = LaqdaInferencer(model_paths=args.model_paths, config=config, device=device_str)
    
    support_text = inferencer.prepare_support_set(args.train_file, labels_dict, config.get('sampler', {}).get('kshot', 5))
    
    print("Iniciando inferência para ensemble (se > 1 modelo)...")
    preds = inferencer.predict_ensemble(test_dataset, support_text, labels_dict, batch_size=32)
    
    # Salva predições (podemos reaproveitar o ensemble saver depois ou apenas gravar bruto aqui)
    from ..inference.ensemble import MajorityVotingEnsemble
    saver = MajorityVotingEnsemble(args.output_dir)
    saver.save(preds)

if __name__ == "__main__":
    main()
