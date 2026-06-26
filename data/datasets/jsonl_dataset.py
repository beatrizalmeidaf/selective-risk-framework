import os
import json
import pandas as pd
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset

class JSONLDataset(Dataset):
    """
    Dataset universal para leitura de arquivos JSONL.
    Padroniza os dados para garantir chaves consistentes ('text', 'label', 'class_id').
    """
    def __init__(self, path: str, class_name_to_id: Dict[str, int] = None, keep_unknown_classes: bool = False):
        """
        Args:
            path: Caminho para o arquivo .jsonl
            class_name_to_id: Dicionário opcional mapeando labels (str) para inteiros (class_id)
            keep_unknown_classes: Se True, mantém classes não mapeadas e atribui id -1 (útil para OOD)
        """
        self.path = path
        self.class_name_to_id = class_name_to_id
        self.keep_unknown_classes = keep_unknown_classes
        
        self.data = self._load_data()
        if not self.data:
            self.df = pd.DataFrame(columns=['text', 'label', 'id', 'class_id'])
            return

        self.df = pd.DataFrame(self.data)
        
        # Padronização de nomes
        if 'sentence' in self.df.columns and 'text' not in self.df.columns:
            self.df['text'] = self.df['sentence']
        if 'class_name' in self.df.columns and 'label' not in self.df.columns:
            self.df['label'] = self.df['class_name']

        self.df = self.df.assign(id=self.df.index.values)

        if self.class_name_to_id is not None and 'label' in self.df.columns:
            if not self.keep_unknown_classes:
                # Filtra apenas dados com labels válidos conhecidos no mapeamento
                self.df = self.df[self.df['label'].isin(self.class_name_to_id.keys())]
            
            # Mapeia e preenche NaN com -1 para classes desconhecidas (se mantidas)
            self.df = self.df.assign(class_id=self.df['label'].map(self.class_name_to_id).fillna(-1).astype(int))
        else:
            self.df = self.df.assign(class_id=-1)

    def _load_data(self) -> list:
        result = []
        if not os.path.exists(self.path):
            print(f"Warning: Dataset file not found at {self.path}")
            return result
            
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """ Retorna o item completo como um dicionário para flexibilidade. """
        return self.df.iloc[idx].to_dict()

    def get_classes(self) -> list:
        """ Retorna lista de classes únicas presentes no dataset. """
        if 'label' in self.df.columns:
            return self.df['label'].unique().tolist()
        if 'class_id' in self.df.columns:
            return self.df['class_id'].unique().tolist()
        return []
