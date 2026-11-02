"""src/models/__init__.py
Fábrica unificada de modelos e contagem de parâmetros.
"""

from typing import Tuple, List, Optional
import torch.nn as nn
from .vit import build_vit_small, get_vit_target_layers, vit_reshape_transform
from .resnet import build_resnet50, get_resnet_target_layers


def build_model(
    architecture: str,
    pretrained: bool = True,
    num_classes: int = 2
) -> nn.Module:
    """
    Fábrica unificada para instanciar as arquiteturas do projeto.
    
    Opções:
      - 'vit_small' ou 'vit': Vision Transformer (vit_small_patch16_224, ~22M parâmetros)
      - 'resnet50' ou 'resnet': CNN (resnet50, ~25.5M parâmetros)
    """
    arch = architecture.lower()
    if arch in ["vit_small", "vit", "vit_small_patch16_224"]:
        return build_vit_small(num_classes=num_classes, pretrained=pretrained)
    elif arch in ["resnet50", "resnet"]:
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Arquitetura '{architecture}' desconhecida. Use 'vit_small' ou 'resnet50'.")


def get_cam_target(architecture: str, model: nn.Module):
    """
    Retorna (target_layers, reshape_transform) para Grad-CAM.
    """
    arch = architecture.lower()
    if arch in ["vit_small", "vit", "vit_small_patch16_224"]:
        return get_vit_target_layers(model), vit_reshape_transform
    elif arch in ["resnet50", "resnet"]:
        return get_resnet_target_layers(model), None
    else:
        raise ValueError(f"Arquitetura '{architecture}' desconhecida.")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Retorna (parâmetros treináveis, total de parâmetros)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
