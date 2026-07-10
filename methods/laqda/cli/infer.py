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
    parser.add_argument(
        '--score_mode',
        type=str,
        default='all',
        choices=['all', 'default', 'margin', 'mcdropout', 'tempscale', 'xmahalanobis'],
        help=(
            'Estratégia de score de confiança a usar na inferência. '
            '"all" executa todas e gera um relatório por estratégia. '
            '"default": -min_dist (original). '
            '"margin": second_min - min_dist. '
            '"mcdropout": Monte Carlo Dropout (N=20 passes). '
            '"tempscale": Temperature Scaling calibrado no val set. '
            '"xmahalanobis": X-Mahalanobis Feature Mixing.'
        )
    )
    parser.add_argument(
        '--mc_passes',
        type=int,
        default=20,
        help='Número de passes estocásticos para MC Dropout (padrão: 20).'
    )
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
    val_dataset = datamodule.valid_dataset  # Usado pelo Temperature Scaling
    
    inferencer = LaqdaInferencer(model_paths=args.model_paths, config=config, device=device_str)
    
    support_text = inferencer.prepare_support_set(
        datamodule.train_dataset.path, labels_dict, config.get('sampler', {}).get('kshot', 5)
    )
    
    print(f"\nIniciando inferência — score_mode='{args.score_mode}'")
    print(f"Output dir: {args.output_dir}\n")

    run_default   = args.score_mode in ('all', 'default')
    run_margin    = args.score_mode in ('all', 'margin')
    run_mcdropout = args.score_mode in ('all', 'mcdropout')
    run_tempscale = args.score_mode in ('all', 'tempscale')
    run_xmaha     = args.score_mode in ('all', 'xmahalanobis')

    if run_default:
        print("=" * 60)
        print(">>> Estratégia: DEFAULT (-min_dist)")
        print("=" * 60)
        inferencer.evaluate_ood(
            test_dataset, support_text, labels_dict,
            batch_size=32, save_dir=args.output_dir
        )

    if run_margin:
        print("=" * 60)
        print(">>> Estratégia: DISTANCE MARGIN (second_min - min_dist)")
        print("=" * 60)
        inferencer.evaluate_ood_margin(
            test_dataset, support_text, labels_dict,
            batch_size=32, save_dir=args.output_dir
        )

    if run_mcdropout:
        print("=" * 60)
        print(f">>> Estratégia: MC DROPOUT ({args.mc_passes} passes)")
        print("=" * 60)
        inferencer.evaluate_ood_mc_dropout(
            test_dataset, support_text, labels_dict,
            batch_size=32, save_dir=args.output_dir,
            n_passes=args.mc_passes
        )

    if run_tempscale:
        print("=" * 60)
        print(">>> Estratégia: TEMPERATURE SCALING (calibrado no val set)")
        print("=" * 60)
        inferencer.evaluate_ood_temp_scaling(
            test_dataset, support_text, labels_dict,
            val_dataset=val_dataset,
            batch_size=32, save_dir=args.output_dir
        )

    if run_xmaha:
        print("=" * 60)
        print(">>> Estratégia: X-MAHALANOBIS (Feature Mixing)")
        print("=" * 60)
        inferencer.evaluate_ood_xmahalanobis(
            test_dataset, support_text, labels_dict,
            batch_size=32, save_dir=args.output_dir
        )

    print("\nInferência concluída!")
    print(f"Relatórios salvos em: {args.output_dir}")
    

if __name__ == "__main__":
    main()
