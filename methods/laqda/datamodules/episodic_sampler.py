import numpy as np
import torch
from typing import List, Iterable
from torch.utils.data import Sampler

class EpisodicKShotSampler(Sampler):
    """
    Sampler episódico para few-shot learning usado no LAQDA.
    Gera tarefas (episodes) contendo um conjunto de suporte e um de consulta.
    """
    def __init__(self, dataset, episodes_per_epoch: int = 100, k: int = 2, n: int = 5, q: int = 25, num_tasks: int = 1, fixed_tasks: List[Iterable[int]] = None, seed: int = None, max_query_total: int = None):
        self.dataset = dataset
        if not hasattr(dataset, 'df'):
            raise ValueError("O dataset fornecido deve conter o atributo 'df' (Dataframe Pandas).")

        self.episodes_per_epoch = episodes_per_epoch
        self.num_tasks = num_tasks
        self.k = k
        self.n = n
        self.q = q
        # Em corpora com muitas classes (nway alto, ex: MMLU 57-way), q por classe
        # fixo multiplica para um episódio gigante (nway * q consultas, codificadas
        # num único forward pass) que estoura a VRAM independente do kshot. Este cap
        # limita o TOTAL de consultas por episódio, reduzindo q por classe quando
        # nway é grande; para corpora com poucas classes (nway * q <= cap) o
        # comportamento é idêntico ao original.
        self.max_query_total = max_query_total
        self.fixed_tasks = fixed_tasks
        self.i_task = 0
        self.seed = seed

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self):
        rs = np.random.RandomState(self.seed)
        for _ in range(self.episodes_per_epoch):
            support_set, query_set = [], []
            # Identifica classes ID válidas (exclui -1 OOD)
            available_classes = [c for c in self.dataset.df['class_id'].unique() if c != -1]
            
            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    k_actual = min(self.k, len(available_classes))
                    episode_classes = rs.choice(available_classes, size=k_actual, replace=False)
                else:
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
                support_k = {k_class: None for k_class in episode_classes}

                if self.max_query_total is not None:
                    effective_q = max(1, min(self.q, self.max_query_total // max(1, len(episode_classes))))
                else:
                    effective_q = self.q

                for k_class in episode_classes:
                    class_samples = df[df['class_id'] == k_class]
                    replace_support = len(class_samples) < self.n
                    support = class_samples.sample(self.n, replace=replace_support, random_state=rs)
                    support_k[k_class] = support
                    for _, s in support.iterrows():
                        support_set.append({"text": s['text'], "label": s['label']})

                for k_class in episode_classes:
                    query_samples_available = df[(df['class_id'] == k_class) & (~df['id'].isin(support_k[k_class]['id']))]
                    if len(query_samples_available) == 0:
                        query = support_k[k_class].sample(effective_q, replace=True, random_state=rs)
                    else:
                        replace_query = len(query_samples_available) < effective_q
                        query = query_samples_available.sample(effective_q, replace=replace_query, random_state=rs)
                    for _, q_row in query.iterrows():
                        query_set.append({"text": q_row['text'], "label": q_row['label']})
                        
            yield support_set, query_set, episode_classes
