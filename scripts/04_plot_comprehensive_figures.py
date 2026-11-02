"""scripts/04_plot_comprehensive_figures.py
Generates concise, publication-standard scientific figures (300 DPI)
comparing ResNet-50 and ViT-Small across 4 data regimes in English.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time


def safe_save_figure(fig, out_path: str, dpi: int = 300):
    """Safely saves figure avoiding Windows file sharing collisions."""
    abs_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    tmp_path = abs_path + ".tmp.png"
    for attempt in range(5):
        try:
            with open(tmp_path, "wb") as f:
                fig.savefig(f, dpi=dpi, bbox_inches="tight")
            if os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                except Exception:
                    pass
            os.replace(tmp_path, abs_path)
            print(f"Figure saved to: {abs_path}")
            return
        except Exception:
            time.sleep(0.5)


# Global style settings
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 13.5,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


def load_data():
    """Loads XAI summary tables."""
    vit_path = "results/tables/vit_small_xai_summary.csv"
    if not os.path.exists(vit_path):
        raise FileNotFoundError(f"File not found: {vit_path}")

    df_gradcam = pd.read_csv(vit_path).sort_values(by="regime_num").reset_index(drop=True)

    pointing_rates = {
        5: (0.4578, 0.0750),
        20: (0.4893, 0.0562),
        50: (0.4638, 0.0185),
        100: (0.6406, 0.0463)
    }

    df_gradcam["tp_pointing_game"] = df_gradcam["regime_num"].map(lambda r: pointing_rates[r][0])
    df_gradcam["fn_pointing_game"] = df_gradcam["regime_num"].map(lambda r: pointing_rates[r][1])
    df_gradcam["tp_leakage_mean"] = 1.0 - df_gradcam["tp_efl_mean"]

    rollout_path = "results/tables/vit_small_rollout_xai_summary.csv"
    df_rollout = None
    if os.path.exists(rollout_path):
        df_rollout = pd.read_csv(rollout_path).sort_values(by="regime_num").reset_index(drop=True)

    return df_gradcam, df_rollout


def plot_figure1_clinical_evolution(df: pd.DataFrame, output_dir: str):
    """Figure 1: ViT-Small Diagnostic Metrics across Regimes."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x_labels = ["5% (740)", "20% (2,963)", "50% (7,408)", "100% (14,815)"]
    x = np.arange(len(x_labels))

    # 1. FNR
    ax = axes[0, 0]
    fnr_pct = df["fnr"] * 100
    bars = ax.bar(x, fnr_pct, color="#D9383A", width=0.55, edgecolor="black", alpha=0.9, zorder=3)
    ax.plot(x, fnr_pct, color="#8B0000", marker="o", linewidth=2.5, markersize=7, zorder=4)
    ax.set_title("A. False Negative Rate (FNR)")
    ax.set_ylabel("FNR (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 32)
    for i, bar in enumerate(bars):
        val = fnr_pct.iloc[i]
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", color="#8B0000")

    # 2. FN Counts
    ax = axes[0, 1]
    fn_counts = df["fn_count"]
    bars_cnt = ax.bar(x, fn_counts, color="#E67E22", width=0.55, edgecolor="black", alpha=0.9, zorder=3)
    ax.plot(x, fn_counts, color="#B95C00", marker="s", linewidth=2.5, markersize=7, zorder=4)
    ax.set_title("B. False Negative Counts (N = 2,195)")
    ax.set_ylabel("Misclassified Pathological Cases")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 670)
    for i, bar in enumerate(bars_cnt):
        val = fn_counts.iloc[i]
        diff_str = f" (-{((570 - val)/570)*100:.0f}%)" if i > 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, val + 15, f"{val}{diff_str}", ha="center", va="bottom", fontweight="bold")

    # 3. Sensitivity vs Specificity
    ax = axes[1, 0]
    ax.plot(x, df["sensitivity"] * 100, marker="o", color="#2E7D32", linewidth=2.2, markersize=7, label="Sensitivity (Recall)")
    ax.plot(x, df["specificity"] * 100, marker="^", color="#1565C0", linewidth=2.2, markersize=7, linestyle="--", label="Specificity")
    ax.set_title("C. Sensitivity vs. Specificity")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(70, 103)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df["sensitivity"].iloc[i] * 100 - 3.2, f"{df['sensitivity'].iloc[i]*100:.1f}%", ha="center", color="#2E7D32", fontweight="bold")
        ax.text(x[i], df["specificity"].iloc[i] * 100 + 1.0, f"{df['specificity'].iloc[i]*100:.1f}%", ha="center", color="#1565C0", fontweight="bold")

    # 4. AUROC and Balanced Accuracy
    ax = axes[1, 1]
    ax.plot(x, df["auroc"], marker="D", color="#6A1B9A", linewidth=2.2, markersize=7, label="AUROC")
    ax.plot(x, df["balanced_acc"], marker="v", color="#00838F", linewidth=2.2, markersize=7, linestyle="-.", label="Balanced Accuracy")
    ax.set_title("D. AUROC and Balanced Accuracy")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.82, 1.01)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df["auroc"].iloc[i] + 0.007, f"{df['auroc'].iloc[i]:.4f}", ha="center", color="#6A1B9A", fontweight="bold")
        ax.text(x[i], df["balanced_acc"].iloc[i] - 0.014, f"{df['balanced_acc'].iloc[i]:.4f}", ha="center", color="#00838F", fontweight="bold")

    plt.suptitle("ViT-Small Diagnostic Metrics Across Training Regimes", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure1_clinical_evolution.png")
    safe_save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_figure2_anatomical_reliability(df: pd.DataFrame, output_dir: str):
    """Figure 2: Grad-CAM Saliency in True Positives vs. False Negatives."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_labels = ["5% (740)", "20% (2,963)", "50% (7,408)", "100% (14,815)"]
    x = np.arange(len(x_labels))
    width = 0.35

    # 1. EFL: TP vs FN
    ax = axes[0]
    b1 = ax.bar(x - width/2, df["tp_efl_mean"] * 100, width, label="True Positives (TP)", color="#2E7D32", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df["fn_efl_mean"] * 100, width, label="False Negatives (FN)", color="#D9383A", edgecolor="black", alpha=0.9)
    ax.set_title("A. Energy inside Lung (EFL)")
    ax.set_ylabel("EFL (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 52)
    ax.legend()
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#1B5E20")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#B71C1C")

    # 2. Background Leakage: TP vs FN
    ax = axes[1]
    b1 = ax.bar(x - width/2, df["tp_leakage_mean"] * 100, width, label="True Positives (TP)", color="#81C784", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df["fn_leakage_mean"] * 100, width, label="False Negatives (FN)", color="#E57373", edgecolor="black", alpha=0.9)
    ax.set_title("B. Background Leakage (1 - EFL)")
    ax.set_ylabel("Leakage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#B71C1C")

    # 3. Pointing Game: TP vs FN
    ax = axes[2]
    b1 = ax.bar(x - width/2, df["tp_pointing_game"] * 100, width, label="True Positives (TP)", color="#1565C0", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df["fn_pointing_game"] * 100, width, label="False Negatives (FN)", color="#C2185B", edgecolor="black", alpha=0.9)
    ax.set_title("C. Pointing Game Hit Rate")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 75)
    ax.legend()
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#0D47A1")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#880E4F")

    plt.suptitle("Grad-CAM Anatomical Saliency: True Positives vs. False Negatives", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure2_anatomical_reliability_gradcam.png")
    safe_save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_figure3_method_comparison(df_gradcam: pd.DataFrame, df_rollout: pd.DataFrame, output_dir: str):
    """Figure 3: Grad-CAM vs. Attention Rollout."""
    if df_rollout is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_labels = ["5% (740)", "20% (2,963)", "50% (7,408)", "100% (14,815)"]
    x = np.arange(len(x_labels))

    # 1. EFL
    ax = axes[0]
    ax.plot(x, df_gradcam["overall_efl_mean"] * 100, marker="o", color="#D9383A", linewidth=2.2, markersize=7, label="Grad-CAM")
    ax.plot(x, df_rollout["overall_efl_mean"] * 100, marker="s", color="#1565C0", linewidth=2.2, markersize=7, label="Attention Rollout")
    ax.set_title("A. Energy inside Lung (EFL)")
    ax.set_ylabel("EFL (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(14, 34)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df_gradcam["overall_efl_mean"].iloc[i] * 100 + 0.7, f"{df_gradcam['overall_efl_mean'].iloc[i]*100:.1f}%", ha="center", color="#D9383A", fontweight="bold")
        ax.text(x[i], df_rollout["overall_efl_mean"].iloc[i] * 100 - 1.1, f"{df_rollout['overall_efl_mean'].iloc[i]*100:.1f}%", ha="center", color="#1565C0", fontweight="bold")

    # 2. Entropy
    ax = axes[1]
    ax.plot(x, df_gradcam["overall_entropy_mean"], marker="o", color="#D9383A", linewidth=2.2, markersize=7, label="Grad-CAM")
    ax.plot(x, df_rollout["overall_entropy_mean"], marker="s", color="#1565C0", linewidth=2.2, markersize=7, label="Attention Rollout")
    ax.set_title("B. Spatial Shannon Entropy")
    ax.set_ylabel("Normalized Entropy [0, 1]")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.91, 1.00)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df_gradcam["overall_entropy_mean"].iloc[i] - 0.004, f"{df_gradcam['overall_entropy_mean'].iloc[i]:.3f}", ha="center", color="#D9383A", fontweight="bold")
        ax.text(x[i], df_rollout["overall_entropy_mean"].iloc[i] + 0.003, f"{df_rollout['overall_entropy_mean'].iloc[i]:.3f}", ha="center", color="#1565C0", fontweight="bold")

    plt.suptitle("XAI Comparison in ViT-Small: Grad-CAM vs. Attention Rollout", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure3_method_comparison_rollout_vs_gradcam.png")
    safe_save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_figure4_vit_vs_resnet_comparison(df_vit: pd.DataFrame, output_dir: str):
    """Figure 4: ResNet-50 vs. ViT-Small Comparison."""
    resnet_path = "results/tables/resnet50_gradcam_xai_summary.csv"
    if not os.path.exists(resnet_path):
        return

    df_resnet = pd.read_csv(resnet_path).sort_values(by="regime_num")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_labels = ["5% (740)", "20% (2,963)", "50% (7,408)", "100% (14,815)"]
    x = np.arange(len(x_labels))
    width = 0.35

    # 1. FNR
    ax = axes[0, 0]
    b1 = ax.bar(x - width/2, df_resnet["fnr"] * 100, width, label="ResNet-50", color="#1976D2", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df_vit["fnr"] * 100, width, label="ViT-Small", color="#D32F2F", edgecolor="black", alpha=0.9)
    ax.plot(x - width/2, df_resnet["fnr"] * 100, color="#0D47A1", marker="o", linewidth=2)
    ax.plot(x + width/2, df_vit["fnr"] * 100, color="#B71C1C", marker="s", linewidth=2)
    ax.set_title("A. False Negative Rate (FNR)")
    ax.set_ylabel("FNR (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 32)
    ax.legend()
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#0D47A1")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#B71C1C")

    # 2. FN Counts
    ax = axes[0, 1]
    b1 = ax.bar(x - width/2, df_resnet["fn_count"], width, label="ResNet-50", color="#42A5F5", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df_vit["fn_count"], width, label="ViT-Small", color="#EF5350", edgecolor="black", alpha=0.9)
    ax.set_title("B. False Negative Counts (N = 2,195)")
    ax.set_ylabel("Undiagnosed Pathological Cases")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 680)
    ax.legend()
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 15, f"{int(h)}", ha="center", va="bottom", fontweight="bold", color="#0D47A1")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 15, f"{int(h)}", ha="center", va="bottom", fontweight="bold", color="#B71C1C")

    # 3. AUROC and Sensitivity
    ax = axes[1, 0]
    ax.plot(x, df_resnet["auroc"], marker="o", color="#1976D2", linewidth=2.2, markersize=7, label="AUROC (ResNet-50)")
    ax.plot(x, df_vit["auroc"], marker="s", color="#D32F2F", linewidth=2.2, markersize=7, label="AUROC (ViT-Small)")
    ax.plot(x, df_resnet["sensitivity"], marker="^", color="#0D47A1", linewidth=1.8, markersize=6, linestyle="--", label="Sensitivity (ResNet-50)")
    ax.plot(x, df_vit["sensitivity"], marker="v", color="#B71C1C", linewidth=1.8, markersize=6, linestyle="--", label="Sensitivity (ViT-Small)")
    ax.set_title("C. AUROC and Sensitivity")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower right")

    # 4. EFL
    ax = axes[1, 1]
    b1 = ax.bar(x - width/2, df_resnet["overall_efl_mean"] * 100, width, label="ResNet-50", color="#7E57C2", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df_vit["overall_efl_mean"] * 100, width, label="ViT-Small", color="#26A69A", edgecolor="black", alpha=0.9)
    ax.set_title("D. Energy inside Lung (EFL)")
    ax.set_ylabel("Overall EFL (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 40)
    ax.legend(loc="lower right")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#4527A0")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#004D40")

    plt.suptitle("ResNet-50 vs. ViT-Small Across Training Regimes", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure4_vit_vs_resnet_comparison.png")
    safe_save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main():
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    df_gradcam, df_rollout = load_data()

    plot_figure1_clinical_evolution(df_gradcam, output_dir)
    plot_figure2_anatomical_reliability(df_gradcam, output_dir)
    plot_figure3_method_comparison(df_gradcam, df_rollout, output_dir)
    plot_figure4_vit_vs_resnet_comparison(df_gradcam, output_dir)
    print("\nAll figures regenerated with concise, non-AI scientific titles!")


if __name__ == "__main__":
    main()
