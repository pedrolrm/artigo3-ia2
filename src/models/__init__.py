"""
Models subpackage for Chest X-Ray classification experiments.

Provides unified model constructors, factories, capacity parity metrics,
and XAI target utilities for ResNet-50 and Vision Transformer architectures
in strict accordance with AGENTS.md guidelines.
"""

from typing import Any, List, Optional, Tuple
import torch.nn as nn

from .resnet import (
    ResNet50,
    build_resnet50,
    get_resnet50,
    get_resnet_target_layer,
    get_resnet_target_layers,
)
from .vit import (
    build_vit_small,
    get_vit_target_layers,
    vit_reshape_transform,
)


def build_model(
    architecture: str,
    pretrained: bool = True,
    num_classes: int = 2,
    drop_rate: float = 0.0,
    **kwargs: Any,
) -> nn.Module:
    """
    Fábrica unificada para instanciar as arquiteturas do projeto com paridade de capacidade.

    Opções de arquitetura:
      - 'resnet50' ou 'resnet': CNN representativa (~25.5M parâmetros) com Stem e forward explícitos
      - 'vit_small', 'vit', ou 'vit_small_patch16_224': Vision Transformer representativo (~22M parâmetros)

    Args:
        architecture (str): Nome da arquitetura desejada.
        pretrained (bool): Se True, carrega pesos ImageNet-1k. Padrão: True.
        num_classes (int): Quantidade de classes diagnósticas (2: Saudável vs Patológico).
        drop_rate (float): Taxa de dropout antes do classificador final. Padrão: 0.0.
        **kwargs: Argumentos adicionais repassados aos construtores específicos.

    Returns:
        nn.Module: Instância do modelo configurada.
    """
    arch = architecture.lower()
    if arch in ["vit_small", "vit", "vit_small_patch16_224"]:
        return build_vit_small(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            **kwargs,
        )
    elif arch in ["resnet50", "resnet"]:
        return build_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Arquitetura '{architecture}' desconhecida. Use 'resnet50' ou 'vit_small'."
        )


def get_cam_target(architecture: str, model: nn.Module) -> Tuple[List[nn.Module], Optional[Any]]:
    """
    Retorna a tupla (target_layers, reshape_transform) para Grad-CAM.

    Args:
        architecture (str): Nome da arquitetura ('resnet50' ou 'vit_small').
        model (nn.Module): Modelo instanciado.

    Returns:
        Tuple[List[nn.Module], Optional[Callable]]:
            - Para ResNet-50: ([model.layer4[-1]], None)
            - Para ViT-Small: ([model.blocks[-1].norm1], vit_reshape_transform)
    """
    arch = architecture.lower()
    if arch in ["vit_small", "vit", "vit_small_patch16_224"]:
        return get_vit_target_layers(model), vit_reshape_transform
    elif arch in ["resnet50", "resnet"]:
        return get_resnet_target_layers(model), None
    else:
        raise ValueError(f"Arquitetura '{architecture}' desconhecida para Grad-CAM.")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Calcula o número de parâmetros do modelo no formato (treináveis, total).

    Args:
        model (nn.Module): Modelo PyTorch.

    Returns:
        Tuple[int, int]: (parâmetros treináveis, total de parâmetros).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


__all__ = [
    # ResNet-50
    "ResNet50",
    "build_resnet50",
    "get_resnet50",
    "get_resnet_target_layer",
    "get_resnet_target_layers",
    # Vision Transformer
    "build_vit_small",
    "get_vit_target_layers",
    "vit_reshape_transform",
    # Unificados
    "build_model",
    "get_cam_target",
    "count_parameters",
]
