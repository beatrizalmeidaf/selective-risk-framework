import argparse
import os
import torch
from data.datamodule import StandardDataModule
from ..datamodules.episodic_sampler import EpisodicKShotSampler
from ..utils.config_loader import load_config
from ..utils.seed import set_seed
from ..modules.laqda_module import LaqdaModule
from ..losses.contrastive_loss import LaqdaContrastiveLoss
from ..trainers.trainer import LaqdaTrainer

def get_parser():
    parser = argparse.ArgumentParser(description="Treinamento LAQDA usando configs YAML")
    parser.add_argument('--config', type=str, default='configs/methods_config.yaml', help='Caminho para o YAML de configuração')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Caminho base do dataset (ex: .../EniacCorpus/few_shot)')
    parser.add_argument('--fold', type=str, required=True, help='Fold (ex: 01)')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Diretório para salvar os modelos')
    parser.add_argument('--epochs', type=int, default=None, help='Número de épocas (sobrescreve o config YAML)')
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
    
    # Sobrescreve epochs se passado na linha de comando
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = args.epochs
    
    set_seed(config.get('hardware', {}).get('seed', 42))
    device = config.get('hardware', {}).get('device', 0)
    device_str = f'cuda:{device}' if torch.cuda.is_available() and device >= 0 else 'cpu'
    print(f"Device: {device_str}")
    
    # Setup Datasets
    datamodule = StandardDataModule(args.dataset_dir, args.fold, keep_unknown_classes=True)
    datamodule.setup()
    labels_dict = datamodule.labels_dict
    
    # CRITICAL FIX: To avoid Train vs Test domain shift in the Label-Aware Encoder,
    # we must train on exactly the same number of classes we will evaluate on.
    total_classes = len(labels_dict)
    config.setdefault('sampler', {})['nway'] = total_classes
    
    sampler_cfg = config.get('sampler', {})
    train_sampler = EpisodicKShotSampler(
        datamodule.train_dataset, 
        episodes_per_epoch=config.get('training', {}).get('episode_train', 100),
        k=sampler_cfg.get('nway', 2),
        n=sampler_cfg.get('kshot', 5),
        q=sampler_cfg.get('qshot', 25)
    )
    
    valid_sampler = None
    if datamodule.valid_dataset is not None:
        valid_sampler = EpisodicKShotSampler(
            datamodule.valid_dataset,
            episodes_per_epoch=config.get('training', {}).get('episode_train', 100),
            k=sampler_cfg.get('nway', 2),
            n=sampler_cfg.get('kshot', 5),
            q=sampler_cfg.get('qshot', 25),
            seed=42 # Fixar semente para validação ser determinística e comparável
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
