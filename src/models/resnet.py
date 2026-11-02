"""src/models/resnet.py
Instanciação e utilitários para a CNN de referência (ResNet-50).
Arquitetura representativa para paridade de capacidade (~25.5M parâmetros).
(Módulo de referência para o parceiro do projeto).
"""

from typing import List
import torch.nn as nn
import timm


def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.0
) -> nn.Module:
    """
    Cria uma instância de ResNet-50 (resnet50) via timm.
    Pesos pré-treinados oficiais do ImageNet-1k.
    """
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    return model


def get_resnet_target_layers(model: nn.Module) -> List[nn.Module]:
    """
    Retorna a camada-alvo para extração de Grad-CAM na ResNet-50.
    Conforme AGENTS.md: model.layer4[-1].
    """
    return [model.layer4[-1]]
