"""src/xai.py
Módulos de Explicabilidade (XAI) para Vision Transformers e CNNs.
Implementa:
  1. ViTAttentionRollout (Abnar & Zuidema 2020) - Método nativo e específico para ViT;
  2. GradCAMGenerator (Grad-CAM padronizado para CNNs e opcional para ViT).
"""

from typing import Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models import get_cam_target


class ViTAttentionRollout:
    """
    Attention Rollout nativo para Vision Transformers (Abnar & Zuidema, ACL 2020).
    Rastreia o fluxo de atenção através de todas as camadas de autoatenção do ViT.
    
    Equação fundamental:
      A_hat_l = 0.5 * A_l + 0.5 * I
      R_L = A_hat_L * A_hat_{L-1} * ... * A_hat_0
      Mapa = R_L[CLS, tokens_espaciais] -> Reshape (14, 14) -> Interpolação (224, 224)
    """

    def __init__(
        self,
        model: nn.Module,
        discard_ratio: float = 0.0,
        head_fusion: str = "mean"
    ):
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.attentions = []
        self.hooks = []

    def _hook(self, module, args):
        """Pre-hook para capturar os pesos de atenção de cada bloco do ViT."""
        x = args[0]
        B, N, C = x.shape
        # Extrai qkv e calcula matriz de atenção sem modificar o fluxo de execução
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        q = q * module.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        self.attentions.append(attn.detach())

    def _register_hooks(self):
        self.attentions = []
        self.hooks = []
        for block in self.model.blocks:
            h = block.attn.register_forward_pre_hook(self._hook)
            self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Executa Attention Rollout para um lote de imagens.
        
        Args:
            input_tensor: Tensor de imagens (B, 3, 224, 224)
            
        Returns:
            np.ndarray de mapas de atenção normalizados [0, 1] de formato (B, 224, 224).
        """
        self.model.eval()
        self._register_hooks()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        self._remove_hooks()

        B = input_tensor.size(0)
        device = input_tensor.device
        num_tokens = self.attentions[0].size(-1)  # 197 para ViT-Small patch 16 (1 CLS + 196 patches)
        eye = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(B, 1, 1)

        result = eye.clone()

        for attn in self.attentions:
            # Fusão de cabeças de atenção
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1)[0]
            elif self.head_fusion == "min":
                attn_fused = attn.min(dim=1)[0]
            else:
                raise ValueError(f"Fusão de cabeças desconhecida: {self.head_fusion}")

            # Descarte opcional de cauda inferior
            if self.discard_ratio > 0.0:
                flat = attn_fused.view(B, -1)
                k = int(flat.size(-1) * self.discard_ratio)
                if k > 0:
                    val, _ = torch.kthvalue(flat, k, dim=-1, keepdim=True)
                    attn_fused = torch.where(attn_fused < val.unsqueeze(-1), torch.zeros_like(attn_fused), attn_fused)

            # Adição da conexão residual e renormalização
            a_hat = 0.5 * attn_fused + 0.5 * eye
            a_hat = a_hat / (a_hat.sum(dim=-1, keepdim=True) + 1e-12)

            result = torch.bmm(a_hat, result)

        # Atenção do token [CLS] (índice 0) para todos os tokens espaciais (índices 1 a 196)
        cls_attention = result[:, 0, 1:]  # (B, 196)
        grid_dim = int(np.sqrt(cls_attention.size(-1)))  # 14
        cls_attention = cls_attention.reshape(B, 1, grid_dim, grid_dim)

        # Interpolação bilinear de alta resolução para 224x224
        heatmaps = F.interpolate(cls_attention, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalização Min-Max em [0, 1] por amostra
        h_min = heatmaps.amin(dim=(1, 2, 3), keepdim=True)
        h_max = heatmaps.amax(dim=(1, 2, 3), keepdim=True)
        heatmaps = (heatmaps - h_min) / (h_max - h_min + 1e-12)

        return heatmaps.squeeze(1).cpu().numpy()


class GradCAMGenerator:
    """
    Gerador uniforme de mapas de ativação Grad-CAM (utilizado para CNNs).
    """

    def __init__(
        self,
        model: nn.Module,
        architecture: str = "resnet50",
        use_cuda: bool = True
    ):
        self.model = model.eval()
        self.architecture = architecture.lower()
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        target_layers, reshape_fn = get_cam_target(self.architecture, self.model)
        self.target_layers = target_layers
        self.reshape_transform = reshape_fn

        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
            reshape_transform=self.reshape_transform
        )

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        targets = None
        if target_class is not None:
            batch_size = input_tensor.size(0)
            targets = [ClassifierOutputTarget(target_class) for _ in range(batch_size)]

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam


def get_xai_explainer(
    model: nn.Module,
    architecture: str = "vit_small",
    method: str = "auto",
    use_cuda: bool = True
):
    """
    Fábrica unificada de explicabilidade:
      - Para ViT: utiliza por padrão Attention Rollout (ou Grad-CAM se explicitado).
      - Para CNN (ResNet-50): utiliza Grad-CAM.
    """
    arch = architecture.lower()
    if "vit" in arch:
        if method in ["auto", "rollout", "attention_rollout"]:
            return ViTAttentionRollout(model)
        else:
            return GradCAMGenerator(model, architecture=arch, use_cuda=use_cuda)
    else:
        return GradCAMGenerator(model, architecture=arch, use_cuda=use_cuda)
