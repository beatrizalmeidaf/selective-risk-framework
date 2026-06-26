from typing import Dict, Any

class BaseDataModule:
    """
    Interface base para DataModules no selective-risk-framework.
    Define os métodos obrigatórios para carregar datasets e configurar samplers.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare_data(self):
        """ Lógica para baixar, indexar ou validar datasets (executado uma vez). """
        pass

    def setup(self):
        """ Inicializa datasets (train, val, test) e samplers (executado em todos os workers). """
        raise NotImplementedError

    def train_dataloader(self):
        """ Retorna o DataLoader de treino. """
        raise NotImplementedError

    def val_dataloader(self):
        """ Retorna o DataLoader de validação. """
        raise NotImplementedError

    def test_dataloader(self):
        """ Retorna o DataLoader de teste. """
        raise NotImplementedError
