import os
import pandas as pd
from typing import Dict, Any
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Dataset universal para leitura de arquivos CSV.
    Padroniza os dados para garantir chaves consistentes ('text', 'label', 'class_id').
    """
    def __init__(self, path: str, class_name_to_id: Dict[str, int] = None, 
                 text_col: str = 'text', label_col: str = 'label'):
        self.path = path
        self.class_name_to_id = class_name_to_id
        
        if not os.path.exists(self.path):
            print(f"Warning: Dataset file not found at {self.path}")
            self.df = pd.DataFrame(columns=['text', 'label', 'id', 'class_id'])
            return

        self.df = pd.read_csv(self.path)
        
        # Padronização
        if text_col in self.df.columns and text_col != 'text':
            self.df['text'] = self.df[text_col]
        if label_col in self.df.columns and label_col != 'label':
            self.df['label'] = self.df[label_col]

        self.df = self.df.assign(id=self.df.index.values)

        if self.class_name_to_id is not None and 'label' in self.df.columns:
            self.df = self.df[self.df['label'].isin(self.class_name_to_id.keys())]
            self.df = self.df.assign(class_id=self.df['label'].map(self.class_name_to_id))
        else:
            self.df = self.df.assign(class_id=-1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.df.iloc[idx].to_dict()

    def get_classes(self) -> list:
        if 'class_id' in self.df.columns:
            return self.df['class_id'].unique().tolist()
        return []
