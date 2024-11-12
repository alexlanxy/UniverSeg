import pathlib
from dataclasses import dataclass
from typing import Literal, Tuple
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
    # Define only two color mappings for binary segmentation
    color_to_label = {
        (0, 0, 0, 255): 0,  # Background class
        (255, 255, 255, 255): 1  # Target class
    }

    seg = PIL.Image.open(path).convert("RGBA")
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    label_mask = np.zeros(seg.shape[:2], dtype=np.int8)

    # Assign label values based on pixel color
    for color, label in color_to_label.items():
        mask = np.all(seg == color, axis=-1)
        label_mask[mask] = label

    label_mask = label_mask[None]


    return label_mask.copy()


from tqdm import tqdm


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    image_path = path / "Image"
    mask_path = path / "Mask"
    print("Absolute path:", mask_path.resolve())

    # Get a sorted list of all .jpg files
    img_files = sorted(image_path.glob("*.jpg"))

    # Use tqdm to add a progress bar to the loop
    for img_file in tqdm(img_files, desc="Processing images and masks"):
        img = process_img(img_file, size=size)

        # Extract identifier to match with mask file
        base_name = img_file.stem.split("_")[1]  # Extract identifier (e.g., "0012169")
        mask_file = mask_path / f"ISIC_{base_name}_segmentation.png"

        # Check if the mask file exists
        if mask_file.exists():
            seg = process_seg(mask_file, size=size)
            data.append((img, seg))
        else:
            print(f"No corresponding mask found for {img_file.name}")

    return data


@dataclass
class ISICDataset(Dataset):
    split: Literal["support", "train", "val"]
    label: int  # Specify target label for binary segmentation
    support_size: int = 5  # Number of support images per target image
    image_size: Tuple[int, int] = (128, 128)

    def __post_init__(self):
        path = pathlib.Path('ISIC2018')
        print("Absolute path:", path.resolve())
        self._data = load_folder(path, size=self.image_size)

        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)

        support_end = int(0.6 * N)
        train_end = support_end + int(0.2 * N)

        self.support_idxs = p[:support_end]
        self.train_idxs = p[support_end:train_end]
        self.val_idxs = p[train_end:]

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

        support_indices = random.sample(list(self.support_idxs), self.support_size)
        support_imgs = []
        support_labels = []
        for s_idx in support_indices:
            s_img, s_seg = self._data[s_idx]
            support_imgs.append(s_img)
            support_labels.append(s_seg)

        # Convert to tensors
        T = torch.from_numpy
        target_img = T(target_img)[None]  # Shape: (1, H, W)
        target_label = T((target_label == self.label).astype(np.float32))  # Shape: (H, W)

        support_imgs = torch.stack([T(img)[None] for img in support_imgs])  # Shape: (support_size, 1, H, W)
        support_labels = torch.stack(
            [T((seg == self.label).astype(np.float32)) for seg in support_labels])  # Shape: (support_size, H, W)

        return target_img, support_imgs, support_labels, target_label
