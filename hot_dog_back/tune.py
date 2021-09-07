import argparse
import logging
import sys

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping

from dataset.data_modules import SeeFoodDataModule
from models.dognet import DogNet


def objective(trial: optuna.Trial):
    seed_everything(420)
    torch.backends.cudnn.benchmark = True

    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
    momemtum = trial.suggest_float('momemtum', 0.9, 1)
    weight_decay = trial.suggest_loguniform('weight_decay', 0.0001, 0.01)
    batch_size = 128

    model = DogNet(lr=lr, momemtum=momemtum, weight_decay=weight_decay)
    datamodule = SeeFoodDataModule(root_dir='data/seefood/', input_size=model.input_size, batch_size=batch_size)
    trainer = pl.Trainer(gpus=[0], max_epochs=100, log_every_n_steps=10, precision=32, num_sanity_val_steps=0,
                         callbacks=[
                             EarlyStopping('val_loss', patience=5, verbose=False),
                             PyTorchLightningPruningCallback(trial, monitor="val_loss")
                         ])
    params = dict(
        lr=lr,
        momemtum=momemtum,
        weight_decay=weight_decay
    )
    trainer.logger.log_hyperparams(params)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning for MaterialNet')
    parser.add_argument('--imp', '-i', action='store_true',
                        help='Run hyperparameters importance analysis')
    parser.add_argument('--name', '-n', required=True, help='Name of the experiment')
    args = parser.parse_args()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study_name = args.name
    save_path = 'optuna_saves'
    storage_name = f'sqlite:///{save_path}/{study_name}.db'

    if args.imp:
        print('Starting hyperparameters importance analysis')
        pruner = optuna.pruners.MedianPruner()
        sampler = optuna.samplers.RandomSampler(seed=420)
        study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name, storage=storage_name,
                                    load_if_exists=True)
        study.optimize(objective, n_trials=20, catch=(RuntimeError,))

        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
    else:
        pruner = optuna.pruners.HyperbandPruner()
        study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
        study.optimize(objective, n_trials=100, catch=(RuntimeError,))

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
