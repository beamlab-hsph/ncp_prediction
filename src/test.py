from pathlib import Path

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from lightning.dataset import NCPDataset
from lightning.model import NCPModel
from lightning.augmentation import get_augmentations
from data import get_test_data


def evaluate_model(X, y, model, augmentations, batch_size, n_cpus, n_gpus):

    # Create test dataset and dataloader
    augm = get_augmentations(augmentations)
    test_ds = NCPDataset(X, y, transforms=augm.test_transforms)
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         num_workers=n_cpus, pin_memory=True, shuffle=False)

    # Need to construct the trainer object to run evaluation
    # To Do: Replace with something more elegant
    trainer = Trainer(gpus=n_gpus)
    
    return trainer.test(model, test_dl)


def evaluate(X, y, augmentations, batch_size, n_cpus, n_gpus, checkpoint_path):
    
    # Load model from checkpoint
    model = NCPModel.load_from_checkpoint(checkpoint_path)

    return evaluate_model()


if __name__ == "__main__":

    import sys
    print(sys.path)

    class Args():
        # Model arguments
        checkpoint_path = '/home/rt156/projects/ncp/models/epoch=28-step=8119.ckpt'
        base_model = 'densenet121'
        batch_size = 192*3
        n_cpus = 6

        # Trainer arguments
        n_gpus = 1

        # Test arguments
        augmentations = None
        data_dir = Path('/n/scratch_gpu/users/r/rt156/penn_processed/')

    args = Args()

    # Get test data
    X, y, z, pt_ids = get_test_data(args.data_dir)
    evaluate(X, y, args.augmentations, args.batch_size, args.n_cpus, args.n_gpus, args.checkpoint_path)
