"""scripts/05_plot_test_evaluation.py
Generates final held-out test set evaluation figures (300 DPI):
  1. results/figures/test_confusion_matrices.png (2x2 Confusion Matrices: 5% vs 100%)
  2. results/figures/test_roc_curves.png (ROC Curves comparing ResNet-50 and ViT-Small)
  3. results/figures/test_probability_distributions.png (Predicted Probability on Pathological Cases)
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


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
    "axes.grid": False
})


def plot_test_confusion_matrices(r5, v5, r100, v100, output_dir: str):
    """Figure: 2x2 Test Set Confusion Matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    classes = ["Normal (0)", "Pathological (1)"]
    models_data = [
        ("A. ResNet-50 (5% Regime)", r5, axes[0, 0]),
        ("B. ViT-Small (5% Regime)", v5, axes[0, 1]),
        ("C. ResNet-50 (100% Regime)", r100, axes[1, 0]),
        ("D. ViT-Small (100% Regime)", v100, axes[1, 1]),
    ]

    for title, df, ax in models_data:
        cm = confusion_matrix(df["y_true"], df["y_pred"])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        ax.set_title(title, pad=12)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        for i in range(len(classes)):
            for j in range(len(classes)):
                count = cm[i, j]
                pct = cm_norm[i, j]
                color = "white" if pct > 55 else "black"
                cell_text = f"{count:,}\n({pct:.1f}%)"
                if i == 1 and j == 0:  # Highlight False Negatives
                    ax.text(j, i, cell_text, ha="center", va="center", color=color, fontweight="bold",
                            bbox=dict(boxstyle="square,pad=0.2", facecolor="red", alpha=0.35, edgecolor="darkred"))
                else:
                    ax.text(j, i, cell_text, ha="center", va="center", color=color, fontweight="bold")

    plt.suptitle("Test Set Confusion Matrices (N = 4,233)", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "test_confusion_matrices.png")
    safe_save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_test_roc_curves(r5, v5, r100, v100, output_dir: str):
    """Figure: Test Set ROC Curves comparing 5% and 100% Regimes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: 5% Regime
    ax = axes[0]
    fpr_r5, tpr_r5, _ = roc_curve(r5["y_true"], r5["y_prob"])
    fpr_v5, tpr_v5, _ = roc_curve(v5["y_true"], v5["y_prob"])
    auc_r5 = roc_auc_score(r5["y_true"], r5["y_prob"])
    auc_v5 = roc_auc_score(v5["y_true"], v5["y_prob"])

    ax.plot(fpr_r5, tpr_r5, color="#1976D2", linewidth=2.5, label=f"ResNet-50 (AUC = {auc_r5:.4f})")
    ax.plot(fpr_v5, tpr_v5, color="#D32F2F", linewidth=2.5, label=f"ViT-Small (AUC = {auc_v5:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")

    # Operating point at 0.5 threshold
    r5_fpr_op = (r5[(r5["y_true"] == 0) & (r5["y_prob"] >= 0.5)].shape[0]) / (r5[r5["y_true"] == 0].shape[0])
    r5_tpr_op = (r5[(r5["y_true"] == 1) & (r5["y_prob"] >= 0.5)].shape[0]) / (r5[r5["y_true"] == 1].shape[0])
    v5_fpr_op = (v5[(v5["y_true"] == 0) & (v5["y_prob"] >= 0.5)].shape[0]) / (v5[v5["y_true"] == 0].shape[0])
    v5_tpr_op = (v5[(v5["y_true"] == 1) & (v5["y_prob"] >= 0.5)].shape[0]) / (v5[v5["y_true"] == 1].shape[0])

    ax.scatter(r5_fpr_op, r5_tpr_op, color="#0D47A1", s=70, zorder=5, marker="o", label=f"ResNet Thresh 0.5 (Sens: {r5_tpr_op*100:.1f}%)")
    ax.scatter(v5_fpr_op, v5_tpr_op, color="#B71C1C", s=80, zorder=5, marker="s", label=f"ViT Thresh 0.5 (Sens: {v5_tpr_op*100:.1f}%)")

    ax.set_title("A. 5% Regime (Scarcity)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel B: 100% Regime
    ax = axes[1]
    fpr_r100, tpr_r100, _ = roc_curve(r100["y_true"], r100["y_prob"])
    fpr_v100, tpr_v100, _ = roc_curve(v100["y_true"], v100["y_prob"])
    auc_r100 = roc_auc_score(r100["y_true"], r100["y_prob"])
    auc_v100 = roc_auc_score(v100["y_true"], v100["y_prob"])

    ax.plot(fpr_r100, tpr_r100, color="#1976D2", linewidth=2.5, label=f"ResNet-50 (AUC = {auc_r100:.4f})")
    ax.plot(fpr_v100, tpr_v100, color="#D32F2F", linewidth=2.5, label=f"ViT-Small (AUC = {auc_v100:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")

    r100_fpr_op = (r100[(r100["y_true"] == 0) & (r100["y_prob"] >= 0.5)].shape[0]) / (r100[r100["y_true"] == 0].shape[0])
    r100_tpr_op = (r100[(r100["y_true"] == 1) & (r100["y_prob"] >= 0.5)].shape[0]) / (r100[r100["y_true"] == 1].shape[0])
    v100_fpr_op = (v100[(v100["y_true"] == 0) & (v100["y_prob"] >= 0.5)].shape[0]) / (v100[v100["y_true"] == 0].shape[0])
    v100_tpr_op = (v100[(v100["y_true"] == 1) & (v100["y_prob"] >= 0.5)].shape[0]) / (v100[v100["y_true"] == 1].shape[0])

    ax.scatter(r100_fpr_op, r100_tpr_op, color="#0D47A1", s=70, zorder=5, marker="o", label=f"ResNet Thresh 0.5 (Sens: {r100_tpr_op*100:.1f}%)")
    ax.scatter(v100_fpr_op, v100_tpr_op, color="#B71C1C", s=80, zorder=5, marker="s", label=f"ViT Thresh 0.5 (Sens: {v100_tpr_op*100:.1f}%)")

    ax.set_title("B. 100% Regime (Full Data)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Test Set ROC Curves: Scarcity vs. Full Data", y=1.00)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "test_roc_curves.png")
    safe_save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def plot_test_probability_distributions(r5, v5, r100, v100, output_dir: str):
    """Figure: Predicted Probability Distributions on Pathological Test Cases."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 35)

    # 5% Regime
    ax = axes[0]
    r5_path = r5[r5["y_true"] == 1]["y_prob"]
    v5_path = v5[v5["y_true"] == 1]["y_prob"]

    ax.hist(r5_path, bins=bins, alpha=0.55, color="#1976D2", label=f"ResNet-50 (FN: 286)", edgecolor="black")
    ax.hist(v5_path, bins=bins, alpha=0.55, color="#D32F2F", label=f"ViT-Small (FN: 570)", edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")
    ax.set_title("A. 5% Regime (Scarcity)")
    ax.set_xlabel("Predicted Probability for Pathological Class")
    ax.set_ylabel("Number of Patients")
    ax.set_xlim([0, 1])
    ax.legend(loc="upper center")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 100% Regime
    ax = axes[1]
    r100_path = r100[r100["y_true"] == 1]["y_prob"]
    v100_path = v100[v100["y_true"] == 1]["y_prob"]

    ax.hist(r100_path, bins=bins, alpha=0.55, color="#1976D2", label=f"ResNet-50 (FN: 169)", edgecolor="black")
    ax.hist(v100_path, bins=bins, alpha=0.55, color="#D32F2F", label=f"ViT-Small (FN: 108)", edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")
    ax.set_title("B. 100% Regime (Full Data)")
    ax.set_xlabel("Predicted Probability for Pathological Class")
    ax.set_ylabel("Number of Patients")
    ax.set_xlim([0, 1])
    ax.legend(loc="upper center")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Predicted Probability Distributions on Pathological Test Cases (Class 1)", y=1.00)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "test_probability_distributions.png")
    safe_save_figure(fig, out_path, dpi=300)
    plt.close(fig)


def main():
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)

    r5 = pd.read_csv("results/tables/resnet50_regime5_gradcam_detailed_test.csv")
    v5 = pd.read_csv("results/tables/vit_small_regime5_gradcam_detailed_test.csv")
    r100 = pd.read_csv("results/tables/resnet50_regime100_gradcam_detailed_test.csv")
    v100 = pd.read_csv("results/tables/vit_small_regime100_gradcam_detailed_test.csv")

    plot_test_confusion_matrices(r5, v5, r100, v100, output_dir)
    plot_test_roc_curves(r5, v5, r100, v100, output_dir)
    plot_test_probability_distributions(r5, v5, r100, v100, output_dir)
    print("\nAll test evaluation figures generated successfully in high resolution (300 DPI)!")


if __name__ == "__main__":
    main()
