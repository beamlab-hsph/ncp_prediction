from torchvision import transforms


class _NormalizeTransform():
    def __init__(self):
        self._mean = [0.485, 0.456, 0.406]
        self._sd = [0.229, 0.224, 0.225]

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=self._mean, std=self._sd)


class NoAugmentations(_NormalizeTransform):
    @property
    def train_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])


class Set1Augmentations(_NormalizeTransform):
    @property
    def train_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            self.normalize_transform
        ])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])


def get_augmentations(name: str):
    if name is None:
        return NoAugmentations()
    elif name == 'Set1':
        return Set1Augmentations()
    else:
        raise ValueError("Expected either None or Set1")
