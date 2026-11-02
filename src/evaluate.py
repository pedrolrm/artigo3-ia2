"""src/evaluate.py
Cálculo de métricas diagnósticas clínicas padronizadas.
Otimizado com inference_mode e suporte a AMP na GPU.
Eixo clínico:
  - Taxa de Falsos Negativos (FNR = FN / (FN + TP) = 1 - Sensibilidade)
  - Sensibilidade (Recall)
  - Especificidade
  - Acurácia Balanceada
  - AUROC
  - Matriz de Confusão (TP, FN, TN, FP)
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score


def compute_diagnostic_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcula as métricas diagnósticas formais com foco em Falsos Negativos.
    Classe 1 = Não Saudável (Positivo clínico)
    Classe 0 = Saudável (Negativo clínico)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Matriz de confusão binária: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Métricas clínicas
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # 1 - Sensibilidade
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5

    return {
        "auroc": float(auroc),
        "fnr": float(fnr),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_acc),
        "fpr": float(fpr),
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "total": int(len(y_true))
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Executa inferência ultrarrápida usando inference_mode e autocast na GPU.
    """
    model.eval()
    all_targets = []
    all_probs = []

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.inference_mode():
        for batch in data_loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1]

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_targets.extend(targets.numpy().tolist() if isinstance(targets, torch.Tensor) else targets)
            all_probs.extend(probs.cpu().float().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    metrics = compute_diagnostic_metrics(y_true, y_prob)
    return metrics, y_true, y_prob
