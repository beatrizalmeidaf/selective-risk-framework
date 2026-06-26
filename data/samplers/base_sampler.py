from torch.utils.data import Sampler

class BaseFrameworkSampler(Sampler):
    """
    Interface base para samplers customizados no framework.
    """
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
