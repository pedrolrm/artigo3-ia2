"""
Módulo da Arquitetura ResNet-50 para Classificação Binária de Radiografias de Tórax.

Especificação Técnica (AGENTS.md):
----------------------------------
- Modelo: ResNet-50 (~25.5 milhões de parâmetros).
- Backbone: Importado de biblioteca pronta (timm / torchvision) com pesos ImageNet-1k.
- Arquitetura Básica Definida:
    * Stem: conv1 (7x7, stride 2), bn1, relu/act1, maxpool (3x3, stride 2)
    * Estágios Convolucionais: layer1, layer2, layer3, layer4
    * Cabeça de Classificação: avgpool (1x1), dropout (opcional), fc (2048 -> num_classes)
- Forward Pass: Definição explícita passo a passo do fluxo de tensores.
- Camada Alvo Grad-CAM: Última camada convolucional (`model.layer4[-1]`).
"""

from typing import Any, List, Optional
import torch
import torch.nn as nn


class ResNet50(nn.Module):
    """
    Arquitetura ResNet-50 para triagem binária de radiografias de tórax.

    Importa o backbone convolucional pronto da ResNet-50 (via timm ou torchvision),
    acopla os estágios convolucionais e define explicitamente o Stem, as camadas,
    a cabeça de classificação (fc) e a passagem direta (forward pass).

    Fluxo de Tensores no Forward:
        1. Entrada:                  [B, 3, 224, 224]
        2. Stem (conv1 + bn + pool): [B, 64, 56, 56]
        3. layer1:                   [B, 256, 56, 56]
        4. layer2:                   [B, 512, 28, 28]
        5. layer3:                   [B, 1024, 14, 14]
        6. layer4 (Grad-CAM alvo):   [B, 2048, 7, 7]
        7. avgpool + flatten:        [B, 2048]
        8. dropout (opcional):       [B, 2048]
        9. fc (classificador):       [B, num_classes] (logits de saída)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        in_chans: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # 1. Carrega o backbone ResNet-50 pré-treinado no ImageNet-1k
        backbone = self._load_backbone(pretrained=pretrained, in_chans=in_chans, **kwargs)

        # 2. Desacopla as camadas estruturais do backbone
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        if hasattr(backbone, "act1"):
            self.relu = backbone.act1
        elif hasattr(backbone, "relu"):
            self.relu = backbone.relu
        else:
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = backbone.maxpool

        # 3. Quatro estágios residuais (layer4[-1] é o alvo oficial do Grad-CAM)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Determina a dimensão de saída da última camada convolucional (2048)
        in_features = 2048
        if hasattr(backbone, "fc") and hasattr(backbone.fc, "in_features"):
            in_features = backbone.fc.in_features
        elif hasattr(backbone, "num_features"):
            in_features = backbone.num_features

        # 4. Cabeça diagnóstica de classificação (Head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

        # Inicialização dos pesos da nova cabeça linear
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    @staticmethod
    def _load_backbone(
        pretrained: bool = True,
        in_chans: int = 3,
        **kwargs: Any,
    ) -> nn.Module:
        """
        Importa o modelo ResNet-50 pronto a partir de timm (especificado no AGENTS.md)
        ou torchvision como fallback.
        """
        # Prioridade 1: timm (padrão oficial definido no AGENTS.md)
        try:
            import timm
            return timm.create_model(
                "resnet50",
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,  # Remove a cabeça original do ImageNet
                **kwargs,
            )
        except ImportError:
            pass

        # Prioridade 2: torchvision.models
        try:
            import torchvision.models as tv_models
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                return tv_models.resnet50(weights=weights)
            except ImportError:
                return tv_models.resnet50(pretrained=pretrained)
        except ImportError:
            raise ImportError(
                "Não foi possível carregar a ResNet-50. "
                "Certifique-se de que 'timm' ou 'torchvision' estejam instalados no ambiente."
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem pelo extrator de características espaciais (Stem e Stages 1 a 4).

        Args:
            x (torch.Tensor): Imagem de entrada no formato [B, 3, 224, 224].

        Returns:
            torch.Tensor: Mapa de ativação da última camada convolucional [B, 2048, 7, 7].
        """
        # Stem Convolucional
        x = self.conv1(x)       # -> [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # -> [B, 64, 56, 56]

        # Estágios Residuais
        x = self.layer1(x)      # -> [B, 256, 56, 56]
        x = self.layer2(x)      # -> [B, 512, 28, 28]
        x = self.layer3(x)      # -> [B, 1024, 14, 14]
        x = self.layer4(x)      # -> [B, 2048, 7, 7] (layer4[-1] é a camada de foco do Grad-CAM)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta (Forward Pass) completa da arquitetura:

        Args:
            x (torch.Tensor): Lote de imagens de entrada [B, 3, 224, 224].

        Returns:
            torch.Tensor: Logits das classes diagnósticas [B, num_classes].
        """
        # 1. Extração de características convolucionais
        x = self.forward_features(x)   # [B, 2048, 7, 7]

        # 2. Pooling global adaptativo
        x = self.avgpool(x)            # [B, 2048, 1, 1]

        # 3. Achatamento para vetor 1D por amostra
        x = torch.flatten(x, 1)        # [B, 2048]

        # 4. Regularização por Dropout
        x = self.drop(x)

        # 5. Classificação Linear
        logits = self.fc(x)            # [B, num_classes]

        return logits


def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    in_chans: int = 3,
    **kwargs: Any,
) -> ResNet50:
    """
    Função fábrica para instanciar a arquitetura ResNet-50.

    Args:
        num_classes (int): Número de classes de saída (2 para Saudável vs Patológico).
        pretrained (bool): Se True, carrega pesos ImageNet-1k. Padrão: True.
        drop_rate (float): Taxa de dropout antes da camada linear. Padrão: 0.0.
        in_chans (int): Número de canais de entrada da imagem. Padrão: 3.
        **kwargs: Parâmetros adicionais repassados para a carga do backbone.

    Returns:
        ResNet50: Modelo configurado com forward explícito e pronto para treino/inferência.
    """
    return ResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        in_chans=in_chans,
        **kwargs,
    )


# Alias para conveniência
get_resnet50 = build_resnet50


def get_resnet_target_layer(model: nn.Module) -> nn.Module:
    """
    Identifica e retorna a camada-alvo para extração de Grad-CAM na ResNet-50.

    Conforme o AGENTS.md (Seção 4), a camada-alvo é a última camada
    convolucional da rede: `model.layer4[-1]`.

    Args:
        model (nn.Module): Instância da ResNet-50.

    Returns:
        nn.Module: Módulo da última camada convolucional (último bloco de layer4).
    """
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise AttributeError("O modelo fornecido não possui o atributo 'layer4'.")


def get_resnet_target_layers(model: nn.Module) -> List[nn.Module]:
    """
    Retorna a camada-alvo em formato de lista [model.layer4[-1]],
    padrão exigido por ferramentas de XAI (ex.: pytorch-grad-cam).

    Args:
        model (nn.Module): Instância da ResNet-50.

    Returns:
        List[nn.Module]: Lista contendo [model.layer4[-1]].
    """
    return [get_resnet_target_layer(model)]
