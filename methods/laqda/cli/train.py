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
    parser.add_argument('--kshot', type=int, default=None, help='K-shot (sobrescreve o config YAML)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate inicial')
    parser.add_argument('--use_sgr', action='store_true', help='Ativa o SGR para travar e salvar o threshold no modelo LAQDA')
    parser.add_argument('--la', type=int, default=None, choices=[0, 1], help='Ablação: liga (1) ou desliga (0) o encoder Label-Aware (sobrescreve o config YAML)')
    parser.add_argument('--disable_transductive_sampler', action='store_true', help='Ablação: desliga o TransductiveQDASampler (aumento de protótipo via vizinhos da query) durante o treino, deixando o protótipo = média pura do suporte')
    parser.add_argument('--sampler_mode', type=str, default=None, choices=['knn', 'sinkhorn'], help='Mecanismo de reestimação de protótipo no treino: "knn" (rho-vizinhos, padrão) ou "sinkhorn" (OT entrópico, fiel ao LAQDA original)')
    parser.add_argument('--rho', type=int, default=None, help='Análise de sensibilidade: tamanho da vizinhança do aumento por consulta (sobrescreve model.k do YAML)')
    return parser

def main():
    # Mitiga fragmentação do alocador CUDA em episódios grandes (corpora com muitas
    # classes, ex: MMLU 57-way), onde blocos de tamanhos variáveis (kshot muda a cada
    # sweep) alocados/liberados repetidamente fragmentam a VRAM e causam OOM mesmo
    # com memória livre agregada suficiente. Precisa ser setado ANTES da primeira
    # alocação CUDA (o allocator só lê a env var na sua inicialização preguiçosa).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Otimização H100: Ativar TensorFloat-32 (TF32) globalmente
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

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
        
    # Sobrescreve kshot se passado na linha de comando
    if args.kshot is not None:
        if 'sampler' not in config:
            config['sampler'] = {}
        config['sampler']['kshot'] = args.kshot
        args.save_dir = os.path.join(args.save_dir, f'kshot_{args.kshot}')
        
    # Sobrescreve LR se passado
    if args.lr is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['learning_rate'] = args.lr

    # Overrides de ablação (Seção de Ablação do paper): variam a arquitetura
    # LAQDA em si, não hiperparâmetros de treino.
    if args.la is not None:
        config.setdefault('model', {})['la'] = args.la
    if args.disable_transductive_sampler:
        config.setdefault('model', {})['use_transductive_sampler'] = False
    if args.sampler_mode is not None:
        config.setdefault('model', {})['sampler_mode'] = args.sampler_mode
    if args.rho is not None:
        config.setdefault('model', {})['k'] = args.rho

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
    # Cap no total de consultas por episódio (nway * q_efetivo). Sem isso, corpora
    # com muitas classes (ex: MMLU, 57-way) multiplicam qshot por nway e geram
    # episódios de milhares de textos que estouram a VRAM — ver
    # methods/laqda/datamodules/episodic_sampler.py. None desativa o cap.
    max_query_total = sampler_cfg.get('max_query_total', 500)
    train_sampler = EpisodicKShotSampler(
        datamodule.train_dataset,
        episodes_per_epoch=config.get('training', {}).get('episode_train', 100),
        k=sampler_cfg.get('nway', 2),
        n=sampler_cfg.get('kshot', 5),
        q=sampler_cfg.get('qshot', 25),
        max_query_total=max_query_total
    )

    valid_sampler = None
    if datamodule.valid_dataset is not None:
        valid_sampler = EpisodicKShotSampler(
            datamodule.valid_dataset,
            episodes_per_epoch=config.get('training', {}).get('episode_train', 100),
            k=sampler_cfg.get('nway', 2),
            n=sampler_cfg.get('kshot', 5),
            q=sampler_cfg.get('qshot', 25),
            max_query_total=max_query_total,
            seed=42 # Fixar semente para validação ser determinística e comparável
        )
        
    # Load global model config
    # LAQDA_ENCODER permite trocar o encoder base sem editar o YAML global,
    # que e' compartilhado por todas as execucoes. Ausente, o comportamento e'
    # identico ao publicado (BERTimbau para pt).
    _enc_override = os.environ.get('LAQDA_ENCODER')
    lang = None  # pode nao ser definido se o YAML global nao existir
    global_config_path = 'configs/model_encoder_config.yaml'
    if os.path.exists(global_config_path):
        global_config = load_config(global_config_path)
        lang = global_config.get('model', {}).get('active_language', 'pt')
        global_model_name = global_config.get('model', {}).get(f'encoder_name_{lang}', 'bert-base-uncased')
    else:
        global_model_name = 'bert-base-uncased'
    # Override so' se a variavel existir; caso contrario o valor do YAML e'
    # preservado. (Uma versao anterior colocava este if ANTES do else, fazendo o
    # else se ligar a ele e forcar bert-base-uncased em toda execucao sem a
    # variavel -- o que treinou modelos com o encoder errado.)
    if _enc_override:
        global_model_name = _enc_override

    # Registrar o encoder no log: sem isto, treinar o encoder errado (p.ex. um
    # corpus em ingles com active_language: "pt") nao deixa nenhum rastro, roda
    # sem erro e so' aparece muito depois na contagem de parametros.
    print(f"Encoder: {global_model_name} "
          f"(origem: {'LAQDA_ENCODER' if _enc_override else f'YAML/{lang}'})", flush=True)

    model_cfg = config.get('model', {})
    
    model = LaqdaModule(
        model_name=global_model_name,
        nway=sampler_cfg.get('nway', 2),
        kshot=sampler_cfg.get('kshot', 5),
        qshot=sampler_cfg.get('qshot', 25),
        la=model_cfg.get('la', 1),
        num_freeze=model_cfg.get('num_freeze', 6),
        k=model_cfg.get('k', 5),
        use_transductive_sampler=model_cfg.get('use_transductive_sampler', True),
        sampler_mode=model_cfg.get('sampler_mode', 'knn')
    )
    model.to(device_str)
    
    loss_fn = LaqdaContrastiveLoss()
    
    trainer = LaqdaTrainer(model, loss_fn, config, device_str)
    trainer.setup_optimizer()
    
    trainer.fit(train_sampler, labels_dict, valid_sampler, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
