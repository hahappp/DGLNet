# Dataset/dataset.py
import os
from typing import List, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image

class FERDataset(Dataset):
    """
    A unified dataset class for Facial Expression Recognition (RAF-DB, FER2013, JAFFE, etc.)
    designed for streamlined and efficient training.
    """
    def __init__(self, images_path: List[str], images_class: List[int], transform: Optional[Callable] = None):
        super(FERDataset, self).__init__()
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        # 1. Load image and convert to RGB format
        img = Image.open(self.images_path[item]).convert('RGB')
        label = self.images_class[item]

        # 2. Apply data augmentations (defined in training script)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function to merge a list of samples into a mini-batch of Tensors.
        """
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels