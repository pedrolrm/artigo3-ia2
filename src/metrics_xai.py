"""src/metrics_xai.py
Métricas quantitativas de explicabilidade e validação anatômica.
Implementa:
  1. Entropia Espacial Normalizada (Dispersão atencional);
  2. Fração de Energia no Pulmão (Energy Fraction inside Lung - EFL);
  3. Vazamento para o Fundo (Background Leakage);
  4. Pointing Game (Hit Rate).
"""

from typing import Dict, Union
import numpy as np
import torch


def compute_spatial_entropy(cam_map: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> float:
    """
    Calcula a Entropia Espacial de Shannon normalizada H(A) no intervalo [0, 1].
    
    H(A) = - 1 / log(N) * sum_{x,y} P(x,y) * log(P(x,y) + eps)
    onde P(x,y) = A(x,y) / sum(A(x,y)).
    """
    if isinstance(cam_map, torch.Tensor):
        cam = cam_map.detach().cpu().numpy()
    else:
        cam = np.array(cam_map, copy=True)

    cam = cam.squeeze()
    if cam.ndim != 2:
        raise ValueError(f"Mapa de ativação esperado 2D (H, W), mas obteve formato: {cam.shape}")

    total_mass = np.sum(cam)
    if total_mass < eps:
        return 1.0  # Ativação uniforme/nula -> máxima incerteza

    p = cam / (total_mass + eps)
    n_pixels = cam.size
    log_n = np.log(n_pixels)

    entropy = -np.sum(p * np.log(p + eps)) / log_n
    return float(np.clip(entropy, 0.0, 1.0))


def compute_efl(
    cam_map: Union[np.ndarray, torch.Tensor],
    lung_mask: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-12
) -> float:
    """
    Calcula a Fração de Energia no Pulmão (Energy Fraction inside Lung - EFL):
    EFL = sum_{(x,y) in M} A(x,y) / sum_{todos (x,y)} A(x,y)
    
    Retorna valor em [0, 1].
    """
    if isinstance(cam_map, torch.Tensor):
        cam = cam_map.detach().cpu().numpy()
    else:
        cam = np.array(cam_map, copy=True)

    if isinstance(lung_mask, torch.Tensor):
        mask = lung_mask.detach().cpu().numpy()
    else:
        mask = np.array(lung_mask, copy=True)

    cam = cam.squeeze()
    mask = (mask.squeeze() > 0.5).astype(np.float32)

    if cam.shape != mask.shape:
        raise ValueError(f"Incompatibilidade de dimensões: CAM {cam.shape} vs Máscara {mask.shape}")

    total_energy = np.sum(cam)
    if total_energy < eps:
        return 0.0

    lung_energy = np.sum(cam * mask)
    efl = lung_energy / (total_energy + eps)
    return float(np.clip(efl, 0.0, 1.0))


def compute_background_leakage(
    cam_map: Union[np.ndarray, torch.Tensor],
    lung_mask: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-12
) -> float:
    """Calcula o vazamento para o fundo: 1 - EFL."""
    efl = compute_efl(cam_map, lung_mask, eps=eps)
    return float(np.clip(1.0 - efl, 0.0, 1.0))


def compute_pointing_game(
    cam_map: Union[np.ndarray, torch.Tensor],
    lung_mask: Union[np.ndarray, torch.Tensor]
) -> int:
    """
    Pointing Game (Hit Rate):
    Retorna 1 se o ponto de pico de ativação argmax(A) estiver dentro da máscara pulmonar,
    caso contrário retorna 0.
    """
    if isinstance(cam_map, torch.Tensor):
        cam = cam_map.detach().cpu().numpy()
    else:
        cam = np.array(cam_map, copy=True)

    if isinstance(lung_mask, torch.Tensor):
        mask = lung_mask.detach().cpu().numpy()
    else:
        mask = np.array(lung_mask, copy=True)

    cam = cam.squeeze()
    mask = (mask.squeeze() > 0.5).astype(np.uint8)

    max_idx = np.unravel_index(np.argmax(cam), cam.shape)
    hit = 1 if mask[max_idx] == 1 else 0
    return hit


def compute_all_xai_metrics(
    cam_map: Union[np.ndarray, torch.Tensor],
    lung_mask: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """Calcula todas as 4 métricas quantitativas de XAI de uma só vez."""
    entropy = compute_spatial_entropy(cam_map)
    efl = compute_efl(cam_map, lung_mask)
    leakage = 1.0 - efl
    pointing_hit = compute_pointing_game(cam_map, lung_mask)

    return {
        "spatial_entropy": entropy,
        "efl": efl,
        "background_leakage": leakage,
        "pointing_game": float(pointing_hit)
    }
