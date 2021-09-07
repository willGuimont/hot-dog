import pathlib
import re

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.data_modules import SeeFoodDataModule


class DogNet(pl.LightningModule):
    def __init__(self, lr=5e-3, momemtum=0.9, weight_decay=0.001):
        super().__init__()
        self.lr = lr
        self.network = models.resnet101(pretrained=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, 2)
        self.input_size = 224
        self.momemtum = momemtum
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momemtum,
                                    weight_decay=self.weight_decay, nesterov=True)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics['val_loss'])

    def forward(self, x):
        x = self.network(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self.network(x)

        loss = F.cross_entropy(x, y)

        self.log('hp_metric', loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self.network(x)

        loss = F.cross_entropy(x, y)
        pred = torch.argmax(x, dim=1)
        acc = (pred == y).float().mean()

        self.log('hp_metric', loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = self.network(x)

        pred = torch.argmax(x, dim=1)
        acc = (pred == y).float().mean()

        self.log('test_acc', acc)

        return acc


if __name__ == '__main__':
    # pytorch setup
    seed_everything(420)
    torch.backends.cudnn.benchmark = True

    # backend
    # lr=0.00216, momemtum=0.9374569, weight_decay=0.00018858
    model = DogNet()

    # data
    datamodule = SeeFoodDataModule('data/seefood/', input_size=model.input_size, batch_size=128)
    # datamodule.visualize(random.choices(list(range(len(datamodule.train_dataset))), k=9))

    # training
    logger = TensorBoardLogger("tb_logs", name="dognet")
    checkpoint_path = pathlib.Path('checkpoints')
    trainer = pl.Trainer(gpus=[0], precision=32, logger=logger, limit_train_batches=1, limit_val_batches=1,
                         min_epochs=0, max_epochs=1000, log_every_n_steps=1,
                         callbacks=[EarlyStopping('val_loss', patience=15, verbose=True),
                                    LearningRateMonitor(),
                                    ModelCheckpoint(
                                        monitor='val_loss',
                                        dirpath=str(checkpoint_path),
                                        filename='material-net-{epoch:02d}-{val_loss:.5f}',
                                        save_top_k=3,
                                        mode='min',
                                    )])
    # trainer.fit(backend, datamodule)
    # trainer.validate()

    # testing
    # trainer.test()

    val_loss_regex = re.compile(r'val_loss=([0-9]+\.?[0-9]*)')
    best_checkpoint = str(min(checkpoint_path.iterdir(), key=lambda x: float(val_loss_regex.search(str(x)).group(1))))
    print(best_checkpoint)
    model.load_from_checkpoint(checkpoint_path=best_checkpoint)

    # good, errors = get_classification_results(backend, datamodule.test_dataset)
    # show_worst(errors)
    # show_best(good)
    # print(get_accuracy(backend, datamodule.test_dataset))

    # continue training
    # trainer = pl.Trainer(resume_from_checkpoint=best_checkpoint, gpus=[0])
