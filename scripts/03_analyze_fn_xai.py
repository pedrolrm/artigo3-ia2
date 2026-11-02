"""scripts/03_analyze_fn_xai.py
Auditoria completa de Explicabilidade (XAI) e Análise da Emergência de Falsos Negativos (FN).
Implementa Attention Rollout nativo para Vision Transformers e Grad-CAM para CNNs.
Otimizado para aceleração via GPU NVIDIA RTX 5060 (8GB).

Gera:
  - Tabelas de auditoria amostra por amostra (results/tables/)
  - Tabela resumo comparativa entre regimes (results/tables/)
  - Gráficos de tendências (FNR, Entropia, EFL, Leakage) (results/figures/)
  - Painéis qualitativos detalhando os Falsos Negativos (results/figures/)
"""

import os
import sys

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.dataset import ChestXRayDataset, IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model
from src.xai import get_xai_explainer
from src.metrics_xai import compute_all_xai_metrics
from src.evaluate import compute_diagnostic_metrics


def configure_hardware():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Desfaz a normalização ImageNet e converte para uint8 [0, 255] RGB."""
    img = tensor.cpu().clone().numpy()
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
    return img


def evaluate_regime_xai(
    architecture: str,
    regime: str,
    test_manifest: str,
    checkpoint_path: str,
    output_tables_dir: str,
    device: torch.device,
    batch_size: int = 32,
    xai_method: str = "rollout"
) -> Dict[str, Any]:
    print(f"\n=================================================================")
    print(f" Avaliação XAI no Test Set | {architecture} | Regime: {regime}%")
    print(f" Método XAI: {xai_method.upper()} | Checkpoint: {checkpoint_path}")
    print(f"=================================================================")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint inexistente: {checkpoint_path}")

    # 1. Carregar modelo e pesos
    model = build_model(architecture, pretrained=False, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Inicializar explicador (Attention Rollout para ViT)
    explainer = get_xai_explainer(model, architecture=architecture, method=xai_method, use_cuda=(device.type == "cuda"))

    # 3. Carregar Test Set com máscaras
    test_ds = ChestXRayDataset(test_manifest, is_train=False, return_mask=True)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    records = []
    all_targets = []
    all_probs = []

    print(f"Processando {len(test_ds):,} radiografias com aceleração de hardware...")

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    for images, targets, masks, paths in tqdm(loader, desc=f"XAI {architecture} {regime}%"):
        batch_size_cur = images.size(0)
        images_device = images.to(device, non_blocking=True)

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(images_device)
            else:
                outputs = model(images_device)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().float().numpy()

        all_targets.extend(targets.numpy().tolist())
        all_probs.extend(probs.tolist())

        # Gerar mapas de explicabilidade
        # Para ViT: Attention Rollout nativo
        # Para CNN: Grad-CAM da classe patológica (classe 1)
        if "vit" in architecture.lower() and hasattr(explainer, "generate"):
            heatmaps = explainer.generate(images_device)
        else:
            heatmaps = explainer.generate(images_device, target_class=1)

        for i in range(batch_size_cur):
            y_true = int(targets[i].item())
            p_val = float(probs[i])
            y_pred = int(p_val >= 0.5)
            cam_map = heatmaps[i]
            mask_np = masks[i].squeeze(0).numpy()

            # Diagnóstico Clínico
            if y_true == 1 and y_pred == 1:
                group = "TP"
            elif y_true == 1 and y_pred == 0:
                group = "FN"  # O erro clínico mais grave
            elif y_true == 0 and y_pred == 0:
                group = "TN"
            else:
                group = "FP"

            # Métricas quantitativas de XAI
            xai_metrics = compute_all_xai_metrics(cam_map, mask_np)

            records.append({
                "architecture": architecture,
                "regime": regime,
                "xai_method": xai_method,
                "image_path": paths[i],
                "y_true": y_true,
                "label": y_true,
                "y_pred": y_pred,
                "y_prob": p_val,
                "group": group,
                "spatial_entropy": xai_metrics["spatial_entropy"],
                "efl": xai_metrics["efl"],
                "background_leakage": xai_metrics["background_leakage"],
                "pointing_game": xai_metrics["pointing_game"]
            })

    detailed_df = pd.DataFrame(records)
    detailed_csv_path = os.path.join(output_tables_dir, f"{architecture}_regime{regime}_{xai_method}_detailed_test.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    # Salva também sem sufixo para compatibilidade
    detailed_df.to_csv(os.path.join(output_tables_dir, f"{architecture}_regime{regime}_detailed_test.csv"), index=False)
    print(f"Tabela detalhada salva em: {detailed_csv_path}")

    # Métricas clínicas globais
    diag_metrics = compute_diagnostic_metrics(all_targets, all_probs)

    # Médias agregadas por grupo diagnóstico
    group_stats = detailed_df.groupby("group").agg({
        "spatial_entropy": ["mean", "std"],
        "efl": ["mean", "std"],
        "background_leakage": ["mean", "std"],
        "pointing_game": ["mean"]
    })
    print("\n--- Estatísticas de XAI por Grupo Diagnóstico ---")
    print(group_stats)

    fn_df = detailed_df[detailed_df["group"] == "FN"]
    tp_df = detailed_df[detailed_df["group"] == "TP"]

    summary = {
        "architecture": architecture,
        "regime": regime,
        "xai_method": xai_method,
        "auroc": diag_metrics["auroc"],
        "fnr": diag_metrics["fnr"],
        "sensitivity": diag_metrics["sensitivity"],
        "specificity": diag_metrics["specificity"],
        "balanced_acc": diag_metrics["balanced_accuracy"],
        "tp_count": diag_metrics["tp"],
        "fn_count": diag_metrics["fn"],
        "tn_count": diag_metrics["tn"],
        "fp_count": diag_metrics["fp"],
        "overall_entropy_mean": detailed_df["spatial_entropy"].mean(),
        "overall_efl_mean": detailed_df["efl"].mean(),
        "overall_leakage_mean": detailed_df["background_leakage"].mean(),
        "overall_pointing_mean": detailed_df["pointing_game"].mean(),
        "fn_entropy_mean": fn_df["spatial_entropy"].mean() if len(fn_df) > 0 else np.nan,
        "fn_efl_mean": fn_df["efl"].mean() if len(fn_df) > 0 else np.nan,
        "fn_leakage_mean": fn_df["background_leakage"].mean() if len(fn_df) > 0 else np.nan,
        "tp_entropy_mean": tp_df["spatial_entropy"].mean() if len(tp_df) > 0 else np.nan,
        "tp_efl_mean": tp_df["efl"].mean() if len(tp_df) > 0 else np.nan,
        "tp_leakage_mean": tp_df["background_leakage"].mean() if len(tp_df) > 0 else np.nan,
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary, detailed_df


def plot_regime_trends(summary_df: pd.DataFrame, output_dir: str, architecture: str, xai_method: str = "rollout"):
    """Gera gráficos de linha mostrando o comportamento das métricas com o aumento de dados."""
    os.makedirs(output_dir, exist_ok=True)
    summary_df = summary_df.sort_values(by="regime_num")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Taxa de Falsos Negativos (FNR)
    axes[0].plot(summary_df["regime_num"], summary_df["fnr"] * 100, marker="o", color="crimson", linewidth=2.5)
    axes[0].set_title(f"Emergência de Falsos Negativos (FNR) - {architecture}")
    axes[0].set_xlabel("% de Dados de Treino")
    axes[0].set_ylabel("FNR (%)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Entropia Espacial
    axes[1].plot(summary_df["regime_num"], summary_df["overall_entropy_mean"], marker="s", color="royalblue", label="Média Geral", linewidth=2)
    if "fn_entropy_mean" in summary_df.columns and not summary_df["fn_entropy_mean"].isna().all():
        axes[1].plot(summary_df["regime_num"], summary_df["fn_entropy_mean"], marker="^", color="darkred", linestyle="--", label="Grupo Falso Negativo (FN)", linewidth=2)
    if "tp_entropy_mean" in summary_df.columns and not summary_df["tp_entropy_mean"].isna().all():
        axes[1].plot(summary_df["regime_num"], summary_df["tp_entropy_mean"], marker="v", color="darkgreen", linestyle="--", label="Grupo Verdadeiro Positivo (TP)", linewidth=2)
    xai_name = "Attention Rollout" if "rollout" in xai_method.lower() else "Grad-CAM"
    axes[1].set_title(f"Dispersão ({xai_name})")
    axes[1].set_xlabel("% de Dados de Treino")
    axes[1].set_ylabel("Entropia Normalizada [0, 1]")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Fração de Energia no Pulmão (EFL) e Leakage
    axes[2].plot(summary_df["regime_num"], summary_df["overall_efl_mean"] * 100, marker="o", color="forestgreen", label="EFL no Pulmão (%)", linewidth=2)
    axes[2].plot(summary_df["regime_num"], summary_df["overall_leakage_mean"] * 100, marker="x", color="purple", label="Vazamento para Fundo (%)", linewidth=2)
    axes[2].set_title("Confiabilidade Anatômica (EFL vs Leakage)")
    axes[2].set_xlabel("% de Dados de Treino")
    axes[2].set_ylabel("Porcentagem (%)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{architecture}_{xai_method}_regime_trends.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Gráfico de tendências salvo em: {plot_path}")


def generate_qualitative_fn_panel(
    detailed_df: pd.DataFrame,
    architecture: str,
    regime: str,
    output_dir: str,
    device: torch.device,
    test_manifest: str = "data/manifests/test.csv",
    xai_method: str = "rollout",
    num_samples: int = 4
):
    """Gera um painel qualitativo visual destacando casos de Falsos Negativos."""
    os.makedirs(output_dir, exist_ok=True)
    fn_samples = detailed_df[detailed_df["group"] == "FN"]
    if len(fn_samples) == 0:
        print(f"[Aviso] Nenhum falso negativo encontrado para o regime {regime}%. Pulando painel qualitativo.")
        return

    test_df = pd.read_csv(test_manifest)
    merged_fn = pd.merge(fn_samples.head(num_samples), test_df[["image_path", "mask_path"]], on="image_path", how="left")
    if "mask_path_y" in merged_fn.columns:
        merged_fn["mask_path"] = merged_fn["mask_path_y"]
    elif "mask_path" not in merged_fn.columns and "mask_path_x" in merged_fn.columns:
        merged_fn["mask_path"] = merged_fn["mask_path_x"]
    merged_fn["label"] = merged_fn["y_true"]
    selected_rows = merged_fn.reset_index(drop=True)

    checkpoint_path = f"results/checkpoints/{architecture}_regime{regime}_best.pth"
    if not os.path.exists(checkpoint_path):
        return

    model = build_model(architecture, pretrained=False, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device).eval()
    explainer = get_xai_explainer(model, architecture=architecture, method=xai_method, use_cuda=(device.type == "cuda"))

    test_ds = ChestXRayDataset(selected_rows, is_train=False, return_mask=True)

    fig, axes = plt.subplots(len(selected_rows), 4, figsize=(16, 4 * len(selected_rows)))
    if len(selected_rows) == 1:
        axes = np.expand_dims(axes, 0)

    for idx in range(len(selected_rows)):
        img_tensor, label, mask_tensor, path = test_ds[idx]
        img_device = img_tensor.unsqueeze(0).to(device)

        if "vit" in architecture.lower() and hasattr(explainer, "generate"):
            cam_map = explainer.generate(img_device)[0]
        else:
            cam_map = explainer.generate(img_device, target_class=1)[0]

        img_np = denormalize_image(img_tensor)
        mask_np = mask_tensor.squeeze().numpy()
        row = selected_rows.iloc[idx]

        # 1. Radiografia Original
        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"Radiografia Patológica (Label 1)\nPred: {row['y_prob']:.2f} (Predito Saudável)")
        axes[idx, 0].axis("off")

        # 2. Máscara Pulmonar Real
        axes[idx, 1].imshow(mask_np, cmap="gray")
        axes[idx, 1].set_title("Máscara Anatômica Pulmonar")
        axes[idx, 1].axis("off")

        # 3. Mapa de Explicabilidade (Rollout ou Grad-CAM)
        xai_name = "Attention Rollout" if "rollout" in xai_method.lower() else "Grad-CAM"
        axes[idx, 2].imshow(img_np)
        axes[idx, 2].imshow(cam_map, cmap="jet", alpha=0.5)
        axes[idx, 2].set_title(f"{xai_name}\nEntropia: {row['spatial_entropy']:.3f}")
        axes[idx, 2].axis("off")

        # 4. Background Leakage / Confiabilidade
        leakage_map = cam_map * (1.0 - mask_np)
        axes[idx, 3].imshow(img_np)
        axes[idx, 3].imshow(leakage_map, cmap="hot", alpha=0.6)
        axes[idx, 3].set_title(f"Vazamento Fora do Pulmão\nLeakage: {row['background_leakage']*100:.1f}% | EFL: {row['efl']*100:.1f}%")
        axes[idx, 3].axis("off")

    plt.suptitle(f"Auditoria de Falsos Negativos (FN) - {architecture} (Regime {regime}%) via {xai_name}", fontsize=15, y=1.00)
    plt.tight_layout()
    out_panel = os.path.join(output_dir, f"{architecture}_regime{regime}_{xai_method}_fn_panel.png")
    plt.savefig(out_panel, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Painel qualitativo de FN salvo em: {out_panel}")


def main():
    parser = argparse.ArgumentParser(description="Auditoria de explicabilidade e Falsos Negativos via Attention Rollout")
    parser.add_argument("--architecture", type=str, default="vit_small", choices=["vit_small", "resnet50"])
    parser.add_argument("--regimes", nargs="+", default=["5", "20", "50", "100"])
    parser.add_argument("--xai_method", type=str, default="rollout", choices=["rollout", "gradcam", "auto"])
    parser.add_argument("--test_manifest", type=str, default="data/manifests/test.csv")
    parser.add_argument("--checkpoints_dir", type=str, default="results/checkpoints")
    parser.add_argument("--tables_dir", type=str, default="results/tables")
    parser.add_argument("--figures_dir", type=str, default="results/figures")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    configure_hardware()
    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de inferência XAI: {device}")

    summaries = []

    for reg in args.regimes:
        ckpt_path = os.path.join(args.checkpoints_dir, f"{args.architecture}_regime{reg}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"[Aviso] Checkpoint {ckpt_path} não encontrado. Pulando regime {reg}%.")
            continue

        summary, detailed_df = evaluate_regime_xai(
            architecture=args.architecture,
            regime=reg,
            test_manifest=args.test_manifest,
            checkpoint_path=ckpt_path,
            output_tables_dir=args.tables_dir,
            device=device,
            batch_size=args.batch_size,
            xai_method=args.xai_method
        )
        summaries.append(summary)

        # Gera painel qualitativo de FN para o regime
        generate_qualitative_fn_panel(
            detailed_df=detailed_df,
            architecture=args.architecture,
            regime=reg,
            output_dir=args.figures_dir,
            device=device,
            test_manifest=args.test_manifest,
            xai_method=args.xai_method
        )

    if len(summaries) > 0:
        summary_df = pd.DataFrame(summaries)
        summary_df["regime_num"] = summary_df["regime"].astype(int)
        summary_csv_path = os.path.join(args.tables_dir, f"{args.architecture}_{args.xai_method}_xai_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        summary_df.to_csv(os.path.join(args.tables_dir, f"{args.architecture}_xai_summary.csv"), index=False)
        print(f"\nTabela consolidada salva em: {summary_csv_path}")

        plot_regime_trends(summary_df, args.figures_dir, args.architecture, xai_method=args.xai_method)
    else:
        print("\nNenhum checkpoint avaliado. Execute o treinamento primeiro via scripts/02_run_experiments.py")


if __name__ == "__main__":
    main()
