import argparse
import os
import torch
from data.datamodule import StandardDataModule
from ..utils.config_loader import load_config
from ..inference.infer import LaqdaInferencer

def get_parser():
    parser = argparse.ArgumentParser(description="Inferência LAQDA")
    parser.add_argument('--config', type=str, default='configs/methods_config.yaml')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Caminho base do dataset')
    parser.add_argument('--fold', type=str, required=True, help='Identificador do fold (ex: 01)')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Caminho(s) para os pesos')
    parser.add_argument('--output_dir', type=str, default='./results', help='Diretório de saída para predições brutas')
    parser.add_argument('--kshot', type=int, default=None, help='K-shot (sobrescreve o config YAML)')
    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config)
    config = config.get('laqda', config)
    
    if args.kshot is not None:
        if 'sampler' not in config:
            config['sampler'] = {}
        config['sampler']['kshot'] = args.kshot
        args.output_dir = os.path.join(args.output_dir, f'kshot_{args.kshot}')
    
    device = config.get('hardware', {}).get('device', 0)
    device_str = f'cuda:{device}' if torch.cuda.is_available() and device >= 0 else 'cpu'
    
    # Prepara datasets
    datamodule = StandardDataModule(args.dataset_dir, args.fold, keep_unknown_classes=True)
    datamodule.setup()
    labels_dict = datamodule.labels_dict
    
    test_dataset = datamodule.test_dataset
    
    inferencer = LaqdaInferencer(model_paths=args.model_paths, config=config, device=device_str)
    
    support_text = inferencer.prepare_support_set(datamodule.train_dataset.path, labels_dict, config.get('sampler', {}).get('kshot', 5))
    
    print("Iniciando inferência para ensemble e geração de métricas OOD...")
    inferencer.evaluate_ood(test_dataset, support_text, labels_dict, batch_size=32, save_dir=args.output_dir)
    
    # Se quiser salvar também as predições de texto bruto, podemos descomentar abaixo:
    # preds = inferencer.predict_ensemble(test_dataset, support_text, labels_dict, batch_size=32)
    # from ..inference.ensemble import MajorityVotingEnsemble
    # saver = MajorityVotingEnsemble(args.output_dir)
    # saver.save(preds)

if __name__ == "__main__":
    main()
