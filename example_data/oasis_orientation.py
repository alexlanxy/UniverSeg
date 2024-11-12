import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import nibabel as nib
import PIL
import torch
from torch.utils.data import Dataset


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    # Load image with PIL directly as it's in .png format
    img = PIL.Image.open(path).convert("L")
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = np.array(img)
    img = img.astype(np.float32) / 255
    # img = np.rot90(img, -1)
    return img.copy()

def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    # Define the RGBA to label mapping
    color_to_label = {
        (0, 0, 0, 255): 0,      # Background
        (85, 85, 85, 255): 1,   # Region 1
        (170, 170, 170, 255): 2, # Region 2
        (255, 255, 255, 255): 3  # Region 3
    }
    
    # Load segmentation mask in RGBA
    seg = PIL.Image.open(path).convert("RGBA")
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    
    # Convert RGBA to integer labels
    seg = np.array(seg)
    label_mask = np.zeros(seg.shape[:2], dtype=np.int8)
    
    # Apply color-to-label mapping
    for color, label in color_to_label.items():
        mask = np.all(seg == color, axis=-1)
        label_mask[mask] = label
    
    # Rotate as necessary
    # label_mask = np.rot90(label_mask, -1)
    return label_mask.copy()




def load_folder(path: pathlib.Path, dataset_name: str, size: Tuple[int, int] = (128, 128)):
    data = []
    image_path = path / dataset_name / "Image"
    mask_path = path / dataset_name / "Mask"
    
    for img_file in sorted(image_path.glob("*.png")):
        img = process_img(img_file, size=size)
        
        # Find the corresponding mask file
        mask_file = mask_path / img_file.name  # Assuming mask filename matches image filename
        seg = process_seg(mask_file, size=size)
        
        data.append((img, seg))
    return data


@dataclass
class OASISDataset(Dataset):
    split: Literal["support", "test"]
    label: int
    orientation: Literal["Sagittal", "Coronal", "Axial"]
    support_frac: float = 0.7

    def __post_init__(self):
        # Define root path to your datasets
        path = pathlib.Path('Oasis_Datasets')
        
        T = torch.from_numpy
        # Load the data for the specified orientation
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(path, self.orientation)]
        
        if self.label is not None:
            self._ilabel = self.label
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = (seg == self._ilabel)[None]
        return img, seg
