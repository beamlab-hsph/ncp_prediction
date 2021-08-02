import numpy as np
from torch.utils.data import Dataset


class NCPDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __getitem__(self, idx):
        image, label = self.X[idx], self.y[idx]
        image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return self.X.shape[0]