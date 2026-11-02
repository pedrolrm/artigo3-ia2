"""src/dataset.py
Dataset PyTorch para radiografias de tórax e respectivas máscaras anatômicas.
Padronizado para resolução 224x224 com alinhamento pixel-a-pixel.
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


# Parâmetros de normalização oficiais do ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool = False, img_size: int = 224):
    """
    Retorna a pipeline de transformações padronizada segundo AGENTS.md:
      - Treino: RandomHorizontalFlip (0.5), RandomRotation (-10, 10), Translação (-5%, 5%), Resize 224, Normalize
      - Val/Teste: Resize 224, ToTensor, Normalize
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ChestXRayDataset(Dataset):
    """
    Dataset para radiografias de tórax com suporte opcional a carregamento de máscara pulmonar.
    
    Atenção estrita ao AGENTS.md:
    As máscaras são redimensionadas com interpolação NEAREST para preservar
    os valores binários {0, 1} sem criar artefatos fracionários nas margens anatômicas.
    """

    def __init__(
        self,
        manifest: Union[str, pd.DataFrame],
        is_train: bool = False,
        return_mask: bool = False,
        img_size: int = 224,
    ):
        if isinstance(manifest, str):
            self.df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        else:
            raise TypeError("Manifest deve ser caminho para CSV ou DataFrame pandas.")

        self.is_train = is_train
        self.return_mask = return_mask
        self.img_size = img_size
        self.transform = get_transforms(is_train=is_train, img_size=img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"]) if "label" in row else int(row.get("y_true", 0))

        # Carrega imagem e replica para 3 canais RGB
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        if not self.return_mask:
            return img_tensor, label

        # Carrega máscara anatômica do pulmão
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = row["mask_path"]
        else:
            mask_path = img_path.replace("\\images\\", "\\masks\\").replace("/images/", "/masks/")
        mask = Image.open(mask_path).convert("L")
        
        # Redimensionamento estrito com NEAREST conforme especificação
        mask_resized = mask.resize((self.img_size, self.img_size), resample=Image.Resampling.NEAREST)
        mask_arr = np.array(mask_resized, dtype=np.float32)
        
        # Binarização estrita: {0.0, 1.0}
        mask_binary = (mask_arr > 127.0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # Shape: (1, 224, 224)

        return img_tensor, label, mask_tensor, img_path
