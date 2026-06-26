import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """
    Define as seeds para todas as bibliotecas de aleatoriedade 
    para garantir reprodutibilidade estrita no framework.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
