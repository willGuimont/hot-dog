from typing import Optional, Callable

from torch.utils.data import Dataset

from dataset.food_dataset import FoodDataset


# TODO to fight class imbalance, try to show hot dog more often / reduce number of not hot dog per epoch
class HotDogDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.food = FoodDataset(root_dir, train)
        self.transform = transform
        self.target_transform = target_transform
        self.hotdog_target = self.food.class_to_target['hot_dog']

    def __len__(self):
        return len(self.food)

    def __getitem__(self, idx: int):
        x, y = self.food[idx]
        y = int(y == self.hotdog_target)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
