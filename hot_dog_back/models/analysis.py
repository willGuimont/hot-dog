import random

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm


class Result:
    def __init__(self):
        self.hot_correct = 0.0
        self.hot_error = 0.0
        self.not_hot_correct = 0.0
        self.not_hot_error = 0.0

    @property
    def tp(self):
        return self.hot_correct

    @property
    def fp(self):
        return self.hot_error

    @property
    def fn(self):
        return self.not_hot_error

    @property
    def tn(self):
        return self.not_hot_correct

    @property
    def precision(self):
        if self.tp + self.fp == 0:
            return 'nan'
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        if self.tp + self.fn == 0:
            return 'nan'
        return self.tp + (self.tp + self.fn)

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        if p == 'nan' or r == 'nan' or p + r == 0:
            return 'nan'
        return 2 * (p * r) / (p + r)

    def __repr__(self):
        return f'TP: {self.tp} \tFP: {self.fp}\n' \
               f'FN: {self.fn} \t TN: {self.tn}\n' \
               f'Precision: {self.precision}\n' \
               f'Recall: {self.recall}\n' \
               f'F1: {self.f1}'


def get_accuracy(model: pl.LightningModule, dataset: Dataset):
    dataloader = DataLoader(dataset, batch_size=64)

    result = Result()
    model = model.to('cuda')

    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        is_correct = targets == predictions

        pred_positive = predictions > 0.5
        pred_negative = predictions < 0.5
        not_correct = torch.logical_not(is_correct)

        result.hot_correct += torch.sum(torch.logical_and(is_correct, pred_positive))
        result.hot_error += torch.sum(torch.logical_and(not_correct, pred_positive))

        result.not_hot_correct += torch.sum(torch.logical_and(is_correct, pred_negative))
        result.not_hot_error += torch.sum(torch.logical_and(not_correct, pred_negative))

    return result


def get_classification_results(model: pl.LightningModule, dataset: Dataset):
    good = []
    errors = []

    dataset = Subset(dataset, range(100))
    dataloader = DataLoader(dataset, batch_size=32)

    for inputs, targets in dataloader:
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        for i, input_ in enumerate(inputs):
            score = probs[i][targets[i]]
            target = targets[i]
            pred = predictions[i]

            input_ = np.transpose(input_, (1, 2, 0))
            input_ = np.asarray(Image.fromarray(np.uint8(input_ * 255)).resize((32, 32)))
            if target == pred:
                good.append((input_, score, target, pred))
            else:
                errors.append((input_, score, target, pred))

    return good, errors


def show_worst(results):
    worst_results = []
    for i, result in enumerate(results):
        if len(worst_results) < 9:
            worst_results.append((result[1], i))
        else:
            if result[1] < worst_results[0][0]:
                worst_results[8] = (result[1], i)

        worst_results.sort()

    imgs, true, pred, score = [], [], [], []
    for i in range(9):
        worst = results[worst_results[i][1]]
        x = np.transpose(worst[0], (0, 1, 2))
        imgs.append(x)
        score.append(worst[1])
        true.append(worst[2])
        pred.append(worst[3])

    imgs = np.asarray(imgs)
    plot_images(imgs, true, cls_pred=pred, score=score, label_names=['hot dog', 'not a hot dog'], title='Worst')


def show_best(results):
    best_results = []
    for i, result in enumerate(results):
        if len(best_results) < 9:
            best_results.append((result[1], i))
        else:
            if result[1] > best_results[0][0]:
                best_results[0] = (result[1], i)

        best_results.sort()

    imgs, true, pred, score = [], [], [], []
    for i in range(8, -1, -1):
        best = results[best_results[i][1]]
        x = np.transpose(best[0], (0, 1, 2))
        imgs.append(x)
        score.append(best[1])
        true.append(best[2])
        pred.append(best[3])

    imgs = np.asarray(imgs)
    plot_images(imgs, true, cls_pred=pred, score=score, label_names=['hot dog', 'not a hot dog'], title='Best')


def show_random(results):
    test = random.sample(results, 9)
    imgs, true, pred = [], [], []
    for i in range(9):
        imgs.append(np.transpose(test[i][0], (1, 2, 0)))
        true.append(test[i][2])
        pred.append(test[i][3])

    imgs = np.asarray(imgs)
    plot_images(imgs, true, cls_pred=pred, label_names=['hot dog', 'not a hot dog'], title='Random')


def plot_images(images, cls_true, *, label_names=None, cls_pred=None, score=None, gray=False, title=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        if gray:
            ax.imshow(images[i], cmap='gray', interpolation='spline16')
        else:
            ax.imshow(images[i, :, :, :], interpolation='spline16')
        # get its equivalent class name

        if label_names:
            cls_true_name = label_names[cls_true[i]]

            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            elif score is None:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            else:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}\nScore: {2:.10f}%".format(cls_true_name, cls_pred_name, score[i] * 100)

            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
