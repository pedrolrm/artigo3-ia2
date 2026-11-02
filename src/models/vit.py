"""src/models/vit.py
Instanciação e utilitários para o Vision Transformer (ViT-Small).
Arquitetura representativa de ViT para paridade de capacidade (~22M parâmetros).
"""

from typing import List
import torch
import torch.nn as nn
import timm


def build_vit_small(
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.0
) -> nn.Module:
    """
    Cria uma instância de ViT-Small (vit_small_patch16_224) via timm.
    Pesos pré-treinados oficiais do ImageNet-1k.
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    return model


def get_vit_target_layers(model: nn.Module) -> List[nn.Module]:
    """
    Retorna a camada-alvo para extração de Grad-CAM no ViT.
    Conforme AGENTS.md: saída do último bloco transformer (model.blocks[-1].norm1).
    """
    return [model.blocks[-1].norm1]


def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """
    Reshape transform para Grad-CAM no ViT:
    1. Descarta o token [CLS] (índice 0);
    2. Converte os 196 tokens espaciais para o grid 2D (height x width = 14 x 14);
    3. Permuta para o formato convencional de mapa de ativação (B, C, H, W).
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


def count_parameters(model: nn.Module) -> int:
    """Retorna o número total de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
