from typing import Optional, Callable

import numpy as np
import torch
from torchvision.datasets import ImageFolder


class SeeFoodDataset(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform, target_transform)

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        y = torch.tensor(np.array(y))
        return x, y
