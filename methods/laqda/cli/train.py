import argparse
import os
import torch
from data.datasets.jsonl_dataset import JSONLDataset
from ..datamodules.episodic_sampler import EpisodicKShotSampler
from ..utils.config_loader import load_config
from ..utils.seed import set_seed
from ..modules.laqda_module import LaqdaModule
from ..losses.contrastive_loss import LaqdaContrastiveLoss
from ..trainers.trainer import LaqdaTrainer

def get_parser():
    parser = argparse.ArgumentParser(description="Treinamento LAQDA usando configs YAML")
    parser.add_argument('--config', type=str, default='configs/methods_config.yaml', help='Caminho para o YAML de configuração')
    parser.add_argument('--train_file', type=str, required=True, help='Caminho para arquivo JSONL de treino')
    parser.add_argument('--valid_file', type=str, help='Caminho para arquivo JSONL de validação')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Diretório para salvar os modelos')
    parser.add_argument('--use_sgr', action='store_true', help='Ativa o SGR para travar e salvar o threshold no modelo LAQDA')
    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config)
    config = config.get('laqda', config)
    
    # Sobrescreve flag use_sgr se passada na linha de comando
    if args.use_sgr:
        if 'metrics' not in config:
            config['metrics'] = {}
        config['metrics']['use_sgr'] = True
    
    set_seed(config.get('hardware', {}).get('seed', 42))
    device = config.get('hardware', {}).get('device', 0)
    device_str = f'cuda:{device}' if torch.cuda.is_available() and device >= 0 else 'cpu'
    print(f"Device: {device_str}")
    
    # Setup Datasets
    temp_dataset = JSONLDataset(args.train_file)
    labels_dict = {str(c): i for i, c in enumerate(temp_dataset.get_classes())}
    train_dataset = JSONLDataset(args.train_file, class_name_to_id=labels_dict)
    
    sampler_cfg = config.get('sampler', {})
    train_sampler = EpisodicKShotSampler(
        train_dataset, 
        episodes_per_epoch=config.get('training', {}).get('episode_train', 100),
        k=sampler_cfg.get('nway', 2),
        n=sampler_cfg.get('kshot', 5),
        q=sampler_cfg.get('qshot', 25)
    )
    
    valid_sampler = None
    if args.valid_file and os.path.exists(args.valid_file):
        valid_dataset = JSONLDataset(args.valid_file, class_name_to_id=labels_dict)
        valid_sampler = EpisodicKShotSampler(
            valid_dataset,
            episodes_per_epoch=config.get('training', {}).get('episode_train', 100),
            k=sampler_cfg.get('nway', 2),
            n=sampler_cfg.get('kshot', 5),
            q=sampler_cfg.get('qshot', 25)
        )
        
    # Load global model config
    global_config_path = 'configs/model_encoder_config.yaml'
    if os.path.exists(global_config_path):
        global_config = load_config(global_config_path)
        lang = global_config.get('model', {}).get('active_language', 'pt')
        global_model_name = global_config.get('model', {}).get(f'encoder_name_{lang}', 'bert-base-uncased')
    else:
        global_model_name = 'bert-base-uncased'

    model_cfg = config.get('model', {})
    
    model = LaqdaModule(
        model_name=global_model_name,
        nway=sampler_cfg.get('nway', 2),
        kshot=sampler_cfg.get('kshot', 5),
        qshot=sampler_cfg.get('qshot', 25),
        la=model_cfg.get('la', 1),
        num_freeze=model_cfg.get('num_freeze', 6),
        k=model_cfg.get('k', 5)
    )
    model.to(device_str)
    
    loss_fn = LaqdaContrastiveLoss()
    
    trainer = LaqdaTrainer(model, loss_fn, config, device_str)
    trainer.setup_optimizer()
    
    trainer.fit(train_sampler, labels_dict, valid_sampler, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
