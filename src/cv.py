from argparse import ArgumentParser
from pathlib import Path

from lightning.model import NCPModel

from pytorch_lightning.loggers import WandbLogger
from data import get_train_data
from train import train_model
from test import evaluate_model
from model_selection import PatientKFold
import pickle


def cross_validate_performance(X, y, pt_ids, args):

    kf = PatientKFold(n_splits=args.cv_splits,
                      shuffle=True, random_state=args.seed)

    perf_metrics_all = []

    for k, (train_index, val_index) in enumerate(kf.split(pt_ids)):

        # Split data
        X_train, y_train, train_pt_ids = X[train_index], y[train_index], pt_ids[train_index]
        X_val, y_val, val_pt_ids = X[val_index], y[val_index], pt_ids[val_index]

        # Construct logger
        wandb_logger = WandbLogger(
            project=args.project_name, entity=args.entity, group=args.group)
        del args.project_name
        del args.entity
        del args.group

        # Fit model
        model = train_model(X_train, y_train, train_pt_ids, base_model=args.base_model, feature_extract=args.feature_extract,
                            use_pretrained=args.use_pretrained, balanced_weights=args.balanced_weights, seed=args.seed,
                            augmentations=args.augmentations, valid_prop=args.valid_prop,
                            batch_size=args.batch_size, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                            early_stopping=args.early_stopping, early_stopping_patience=args.early_stopping_patience,
                            fast_dev_run=args.fast_dev_run, limit_train_batches=args.limit_train_batches,
                            limit_val_batches=args.limit_val_batches, max_epochs=args.max_epochs, min_epochs=args.min_epochs,
                            model_save_dir=args.model_save_dir, profiler=args.profiler, logger=wandb_logger)

        # Evaluate model
        perf_metrics = evaluate_model(X_val, y_val, model, augmentations=args.augmentations,
                                      batch_size=args.batch_size, n_cpus=args.n_cpus, n_gpus=args.n_gpus)
        perf_metrics_all.append(perf_metrics)

    with open('data/cv_metrics.pickle', 'wb') as handle:
        pickle.dump(perf_metrics_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add program args
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--data_dir', type=Path,
                        default='/n/scratch_gpu/users/r/rt156/tufts_processed/')
    parser.add_argument('--model_save_dir', type=Path,
                        default='~/projects/ncp/models/')
    parser.add_argument('--cv_splits', type=int,
                        default=5)

    # Add wandb specific args
    parser.add_argument('--project_name', type=str,
                        default='deep_ncp_prediction')
    parser.add_argument('--entity', type=str, default='beamlab')
    parser.add_argument('--group', type=str, default='cv')

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

    # Evaluate
    cross_validate_performance(X, y, pt_ids, args)
