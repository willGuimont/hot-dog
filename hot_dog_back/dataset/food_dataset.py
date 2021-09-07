import json
import pathlib
from typing import Optional, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        possible_targets, samples = self.__get_samples(root_dir, train)

        self.class_to_target = dict(zip(possible_targets, range(len(possible_targets))))
        self.target_to_class = {v: k for (k, v) in self.class_to_target.items()}
        self.paths = [path for (path, _) in samples]
        self.targets = [self.class_to_target[label] for (_, label) in samples]
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def __get_samples(root_dir, train):
        root_dir = pathlib.Path(root_dir)
        json_descriptor_path = root_dir.joinpath('meta').joinpath('train.json' if train else 'test.json')
        descriptor: dict = json.load(open(json_descriptor_path, 'r'))
        image_path = root_dir.joinpath('images')
        samples = [(str(image_path.joinpath(path)) + '.jpg', label) for (label, paths) in descriptor.items() for path in
                   paths]
        possible_targets = list(descriptor.keys())
        return possible_targets, samples

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path, label = self.paths[idx], self.targets[idx]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, torch.tensor(np.array(label))
