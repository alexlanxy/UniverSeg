import pathlib
from dataclasses import dataclass
from typing import Literal, Tuple, List
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import random

def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path).convert("L")
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255
    return img.copy()

def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    color_to_label = {
        (0, 0, 0, 255): 0,
        (85, 85, 85, 255): 1,
        (170, 170, 170, 255): 2,
        (255, 255, 255, 255): 3
    }
    seg = PIL.Image.open(path).convert("RGBA")
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    label_mask = np.zeros(seg.shape[:2], dtype=np.int8)
    for color, label in color_to_label.items():
        mask = np.all(seg == color, axis=-1)
        label_mask[mask] = label
    return label_mask.copy()

def load_folder(path: pathlib.Path, dataset_name: str, size: Tuple[int, int] = (128, 128)):
    data = []
    image_path = path / dataset_name / "Image"
    mask_path = path / dataset_name / "Mask"
    
    for img_file in sorted(image_path.glob("*.png")):
        img = process_img(img_file, size=size)
        mask_file = mask_path / img_file.name
        seg = process_seg(mask_file, size=size)
        data.append((img, seg))
    return data

@dataclass
class UniverSegDataset(Dataset):
    split: Literal["support", "train", "val"]
    orientation: Literal["Sagittal"]
    label: int  # Specify target label for binary segmentation
    support_size: int = 5  # Number of support images per target image
    image_size: Tuple[int, int] = (128, 128)

    def __post_init__(self):
        path = pathlib.Path(__file__).parent / 'Oasis_Datasets'
        print("Absolute path:", path.resolve())
        self._data = load_folder(path, self.orientation, size=self.image_size)
        
        # Create index splits for support, train, and val
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        
        # Calculate support, train, and validation splits
        support_end = int(0.6 * N)
        train_end = support_end + int(0.2 * N)
        
        # Define indices for each split
        self.support_idxs = p[:support_end]
        self.train_idxs = p[support_end:train_end]
        self.val_idxs = p[train_end:]
        
        # Set the current split indices based on `split`
        if self.split == "support":
            self._idxs = self.support_idxs
        elif self.split == "train":
            self._idxs = self.train_idxs
        elif self.split == "val":
            self._idxs = self.val_idxs
        else:
            raise ValueError("Invalid split type. Choose from 'support', 'train', or 'val'.")

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        target_idx = self._idxs[idx]
        target_img, target_label = self._data[target_idx]
        
        # Select random support samples only from the support set
        support_indices = random.sample(list(self.support_idxs), self.support_size)
        support_imgs = []
        support_labels = []
        for s_idx in support_indices:
            s_img, s_seg = self._data[s_idx]
            support_imgs.append(s_img)
            support_labels.append(s_seg)
        
        # Convert to tensors
        T = torch.from_numpy
        target_img = T(target_img)[None]  # Add channel dimension
        target_label = T((target_label == self.label).astype(np.float32))  # Binarize target label
        
        support_imgs = torch.stack([T(img)[None] for img in support_imgs])  # Shape: (support_size, 1, H, W)
        support_labels = torch.stack([T((seg == self.label).astype(np.float32)) for seg in support_labels])  # Binarize support labels

        return target_img, support_imgs, support_labels, target_label
