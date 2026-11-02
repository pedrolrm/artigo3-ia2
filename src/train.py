"""src/train.py
Pipeline de treinamento de alta performance para Vision Transformers e CNNs.
Otimizado especificamente para arquiteturas modernas NVIDIA (RTX 5060 8GB):
  - Aceleração com Tensor Cores via TF32 (allow_tf32 = True)
  - Autocast com bfloat16 nativo (sem perda de precisão ou risco de underflow)
  - Pinned Memory e transferências assíncronas non_blocking=True
  - DataLoader com persistent_workers
  - Early stopping estrito baseado no AUROC de validação (paciência de 7 épocas)
"""

import os
import time
import argparse
import random
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.dataset import ChestXRayDataset
from src.models import build_model, count_parameters
from src.evaluate import evaluate_model


def configure_hardware_optimizations(device: torch.device):
    """
    Ativa otimizações de ponta para a GPU NVIDIA RTX 5060 (Blackwell):
    - TF32 para multiplicação de matrizes e convoluções
    - Otimizador de kernels cuDNN Benchmark
    """
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Limpa resíduos de memória VRAM
        torch.cuda.empty_cache()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True
) -> float:
    model.train()
    total_loss = 0.0

    # Determina suporte a bfloat16 para os Tensor Cores da RTX 5060
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def train_model(
    architecture: str = "vit_small",
    regime: str = "5",
    manifests_dir: str = "data/manifests",
    results_dir: str = "results",
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 3,
    patience: int = 7,
    seed: int = 42,
    num_workers: int = 4,
    use_amp: bool = True
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_hardware_optimizations(device)

    print(f"\n=================================================================")
    print(f" Treinamento Otimizado: {architecture} | Regime: {regime}%")
    print(f" Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f" VRAM Disponível: {vram_gb:.2f} GB | Aceleração AMP (bfloat16): {use_amp} | TF32: Ativo")
    print(f"=================================================================")

    # 1. Carregar Manifestos
    train_manifest = os.path.join(manifests_dir, f"train_{regime}.csv")
    val_manifest = os.path.join(manifests_dir, "val.csv")

    if not os.path.exists(train_manifest) or not os.path.exists(val_manifest):
        raise FileNotFoundError(f"Manifesto não encontrado: {train_manifest} ou {val_manifest}")

    train_ds = ChestXRayDataset(train_manifest, is_train=True, return_mask=False)
    val_ds = ChestXRayDataset(val_manifest, is_train=False, return_mask=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )

    print(f"Amostras de Treino: {len(train_ds):,} | Validação: {len(val_ds):,} | Lote: {batch_size}")

    # 2. Instanciar Modelo
    model = build_model(architecture, pretrained=True, num_classes=2)
    trainable_p, total_p = count_parameters(model)
    print(f"Modelo: {architecture} | Parâmetros Treináveis: {trainable_p:,} / Total: {total_p:,}")
    model = model.to(device)

    # 3. Otimizador AdamW e Scheduler Cosine com Warmup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Diretórios de resultados
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    best_val_auroc = -1.0
    patience_counter = 0
    best_epoch = 0
    history = []

    model_tag = f"{architecture}_regime{regime}"
    best_checkpoint_path = os.path.join(checkpoints_dir, f"{model_tag}_best.pth")
    log_csv_path = os.path.join(logs_dir, f"{model_tag}_history.csv")

    print("\nIniciando épocas de treino...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=use_amp)
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        scheduler.step()
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        val_auroc = val_metrics["auroc"]
        val_fnr = val_metrics["fnr"]
        val_acc = val_metrics["balanced_accuracy"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_auroc,
            "val_fnr": val_fnr,
            "val_balanced_acc": val_acc,
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "lr": current_lr,
            "time_sec": epoch_time
        })

        print(
            f"Época {epoch:02d}/{num_epochs:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val FNR: {val_fnr*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Tempo: {epoch_time:.1f}s"
        )

        # Early Stopping com paciência de 7 épocas
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "architecture": architecture,
                "regime": regime,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": val_auroc,
                "val_metrics": val_metrics,
            }, best_checkpoint_path)
            print(f"  ==> [MELHOR CHECKPOINT] Val AUROC: {val_auroc:.4f} salvo!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Paciência de {patience} épocas atingida na época {epoch}.")
                break

    # Persiste log
    pd.DataFrame(history).to_csv(log_csv_path, index=False)
    print(f"\nTreinamento concluído!")
    print(f"Melhor AUROC: {best_val_auroc:.4f} na época {best_epoch}")
    print(f"Arquivo de pesos: {best_checkpoint_path}")
    print(f"Histórico salvo: {log_csv_path}")

    # Libera memória na GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "checkpoint_path": best_checkpoint_path,
        "log_path": log_csv_path
    }


def main():
    parser = argparse.ArgumentParser(description="Treinamento otimizado para RTX 5060")
    parser.add_argument("--architecture", type=str, default="vit_small", choices=["vit_small", "resnet50"])
    parser.add_argument("--regime", type=str, default="5", choices=["5", "20", "50", "100"])
    parser.add_argument("--manifests_dir", type=str, default="data/manifests")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true", help="Desativa bfloat16 mixed precision")
    args = parser.parse_args()

    train_model(
        architecture=args.architecture,
        regime=args.regime,
        manifests_dir=args.manifests_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        use_amp=(not args.no_amp)
    )


if __name__ == "__main__":
    main()
