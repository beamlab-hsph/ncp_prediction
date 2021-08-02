from typing import Dict
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from data import get_train_data
from utils import get_class_weights
from model_selection import patient_train_test_split
from lightning.dataset import NCPDataset
from lightning.model import NCPModel
from lightning.augmentation import get_augmentations


def train_model(X, y, pt_ids, base_model, feature_extract, use_pretrained, balanced_weights, seed, augmentations, valid_prop,
                batch_size, num_cpus, num_gpus, early_stopping, early_stopping_patience, fast_dev_run, limit_train_batches,
                limit_val_batches, max_epochs, min_epochs, model_save_dir, profiler, logger=None):
    callbacks = []

    # Make everything deterministic
    seed_everything(seed, workers=True)

    # Construct model from params
    model = NCPModel(
        base_model,
        feature_extract,
        use_pretrained,
        pos_weight=get_class_weights(y)[1] if balanced_weights else None
    )

    # Create dataset and dataloader objects
    augm = get_augmentations(augmentations)

    if valid_prop:
        if early_stopping:

            callbacks.append(EarlyStopping(
                monitor="valid_AUROC_epoch", patience=early_stopping_patience, mode="max"))

        X_train, X_val, y_train, y_val = patient_train_test_split(
            X, y, pt_ids, test_size=valid_prop)

        val_ds = NCPDataset(X_val, y_val, transforms=augm.test_transforms)
        val_dl = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=num_cpus, pin_memory=True, shuffle=False)
        train_ds = NCPDataset(
            X_train, y_train, transforms=augm.train_transforms)
        train_dl = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=num_cpus, pin_memory=True, shuffle=True)
    else:
        val_dl = None
        train_ds = NCPDataset(X, y, transforms=augm.train_transforms)
        train_dl = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=num_cpus, pin_memory=True, shuffle=True)

    # Model checkpoint
    if model_save_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_save_dir,
            save_top_k=1,
            verbose=True,
            monitor='valid_AUROC_epoch',
            mode='max'
        )
        callbacks.append(checkpoint_callback)

    # Construct trainer object
    trainer = Trainer(
        profiler=profiler,
        accelerator='ddp' if num_gpus > 1 else None,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        gpus=num_gpus,
        logger=logger,
        deterministic=True,
        log_every_n_steps=50,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        num_sanity_val_steps=0,  # hack to disable logging of validation sanity check,
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

    if early_stopping:
        best_model = NCPModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        return best_model
    else:
        return model


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add program args
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--data_dir', type=Path,
                        default='/n/scratch_gpu/users/r/rt156/tufts_processed/')
    parser.add_argument('--model_save_dir', type=Path,
                        default='~/projects/ncp/models/')

    # Add wandb specific args
    parser.add_argument('--project_name', type=str,
                        default='deep_ncp_prediction')
    parser.add_argument('--entity', type=str, default='beamlab')

    # Add model args
    parser = NCPModel.add_model_specific_args(parser)
    parser.add_argument('--augmentations', type=str, default='Set1')
    parser.add_argument('--balanced_weights', type=bool, default=True)

    # Add training args
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--min_epochs', type=int, default=1)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_cpus', type=int, default=6)
    parser.add_argument('--profiler', type=str, default=None)
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)

    args = parser.parse_args()

    # Get training data
    X, y, z, pt_ids = get_train_data(args.data_dir)
    del args.data_dir

    # Construct logger
    wandb_logger = WandbLogger(project=args.project_name, entity=args.entity)
    del args.project_name
    del args.entity

    # Train model
    train_model(X, y, pt_ids, **vars(args), logger=wandb_logger)
