import pathlib

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader

import pipeline.pipeline as pp
from dataset.seefood_dataset import SeeFoodDataset
from dataset.unnormalize import UnNormalize
from models.analysis import plot_images


class SeeFoodDataModule(pl.LightningDataModule):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root_dir: str, input_size: int, batch_size: int = 32,
                 num_workers: int = 8, pin_memory: bool = True):
        super().__init__()
        root_dir = pathlib.Path(root_dir)
        target_transform = pp.Lambda(lambda y: 1 - y)
        self.train_dataset = SeeFoodDataset(str(root_dir.joinpath('train')), pp.Compose([
            T.ColorJitter(),
            T.RandomAffine(degrees=1, translate=(0.05, 0.05),
                           shear=(-1, 1, -1, 1), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ]), target_transform)
        self.test_dataset = SeeFoodDataset(str(root_dir.joinpath('test')), pp.Compose([
            T.Scale(input_size),
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ]), target_transform)
        self.unorm = UnNormalize(self.mean, self.std)
        self.train_split, self.val_split = self.__get_split(self.train_dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @staticmethod
    def __get_split(dataset):
        num_sample = len(dataset)
        train_percent = 0.8
        num_train = int(np.ceil(num_sample * train_percent))
        num_valid = num_sample - num_train
        return random_split(dataset, [num_train, num_valid])

    def visualize(self, idx: [int]):
        imgs = []
        cls_true = []
        for i in idx:
            img, label = self.train_dataset[i]
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            imgs.append(img)
            cls_true.append(label.item())
        imgs = np.asarray(imgs)
        plot_images(imgs, cls_true, label_names=['not a hot dog', 'hot dog'], title='Data augmentation')

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)
