import os
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torch
from typing import Any, Tuple

class MyClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed = self.transforms(image=np.array(image), mask=np.array(target))
            return transformed['image'], transformed['mask']
        return image, target

def get_transforms():
    """Define data augmentations and normalization."""
    return A.Compose([
        A.Resize(256, 512),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_dataloaders(data_dir='/kaggle/input/cityscapes/cityscapes', batch_size=16, num_workers=4):
    """Create train, validation, and test dataloaders."""
    transform = get_transforms()
    
    train_dataset = MyClass(data_dir, split='train', mode='fine', target_type='semantic', transforms=transform)
    val_dataset = MyClass(data_dir, split='val', mode='fine', target_type='semantic', transforms=transform)
    test_dataset = MyClass(data_dir, split='val', mode='fine', target_type='semantic', transforms=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader