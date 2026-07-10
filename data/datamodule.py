import os
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from data.datasets.jsonl_dataset import JSONLDataset

class StandardDataModule:
    """
    DataModule padronizado para carregamento de dados em splits definidos.
    Evita data leakage forçando o carregamento estruturado e o 'fit' 
    do dicionário de classes apenas no conjunto de treino.
    """
    def __init__(self, dataset_dir: str, fold: str, batch_size: int = 16, keep_unknown_classes: bool = False, kshot: int = None):
        self.dataset_dir = dataset_dir
        self.fold = str(fold).zfill(2) if isinstance(fold, int) or (isinstance(fold, str) and len(fold) == 1) else str(fold)
        self.batch_size = batch_size
        self.keep_unknown_classes = keep_unknown_classes
        self.kshot = kshot
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.labels_dict = None

    def setup(self):
        fold_dir = os.path.join(self.dataset_dir, self.fold)
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

        train_file = os.path.join(fold_dir, "train.jsonl")
        valid_file = os.path.join(fold_dir, "valid.jsonl")
        test_file = os.path.join(fold_dir, "test.jsonl")

        # 1. Fit no conjunto de treino
        temp_train = JSONLDataset(train_file)
        all_classes = temp_train.get_classes()
        
        # Tentar carregar OOD splits
        import json
        corpus_name = os.path.basename(os.path.dirname(self.dataset_dir.rstrip('/')))
        splits_file = "configs/ood_splits.json"
        
        id_classes = all_classes
        if os.path.exists(splits_file):
            with open(splits_file, 'r', encoding='utf-8') as f:
                splits = json.load(f)
                if corpus_name in splits and self.fold in splits[corpus_name]:
                    id_classes = splits[corpus_name][self.fold]["id_classes"]

        # Mapeia apenas as classes ID (0 a K-1)
        self.labels_dict = {str(c): i for i, c in enumerate(id_classes)}

        # 2. Instanciar os datasets aplicando o fit do treino
        # keep_unknown_classes=False garante que o TREINO elimine as classes OOD
        self.train_dataset = JSONLDataset(train_file, class_name_to_id=self.labels_dict, keep_unknown_classes=False)
        
        if self.kshot is not None:
            df = self.train_dataset.df
            self.train_dataset.df = df.groupby('class_id', group_keys=False).apply(lambda x: x.sample(n=min(len(x), self.kshot), random_state=42)).reset_index(drop=True)
        
        if os.path.exists(valid_file):
            # Validação também é ESTRITAMENTE ID-only.
            # Motivo: usar OOD na validacao = data leakage do scorer OOD,
            # pois o early stopping e a selecao do threshold veem a distribuicao OOD
            # antes da avaliacao final. Protocolo padrao (OpenOOD, ODIN, Energy Score).
            self.valid_dataset = JSONLDataset(valid_file, class_name_to_id=self.labels_dict, keep_unknown_classes=False)
            
        if os.path.exists(test_file):
            # Teste retém OOD classes (como -1) se self.keep_unknown_classes for True
            self.test_dataset = JSONLDataset(test_file, class_name_to_id=self.labels_dict, keep_unknown_classes=self.keep_unknown_classes)

    def train_dataloader(self, collate_fn=None, shuffle=True) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    def val_dataloader(self, collate_fn=None) -> Optional[DataLoader]:
        if self.valid_dataset is None: return None
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self, collate_fn=None) -> Optional[DataLoader]:
        if self.test_dataset is None: return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
