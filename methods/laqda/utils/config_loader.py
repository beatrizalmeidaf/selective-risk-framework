import yaml

def load_config(config_path: str) -> dict:
    """
    Carrega o arquivo de configuração YAML.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
