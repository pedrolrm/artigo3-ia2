"""scripts/04_plot_comprehensive_figures.py
Gera figuras científicas consolidadas de alta resolução (300 DPI)
para o artigo científico, retratando a evolução clínica e anatômica
do Vision Transformer (ViT-Small) em 4 regimes de dados.

Gera:
  1. results/figures/figure1_clinical_evolution.png (Métricas Clínicas e Falsos Negativos)
  2. results/figures/figure2_anatomical_reliability_gradcam.png (EFL, Leakage e Pointing Game: TP vs FN)
  3. results/figures/figure3_method_comparison_rollout_vs_gradcam.png (Comparativo Grad-CAM vs Rollout)
"""

import os
import sys

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Configurações globais de estilo para padrão de publicação científica
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


def load_data():
    gradcam_summary_path = "results/tables/vit_small_gradcam_xai_summary.csv"
    rollout_summary_path = "results/tables/vit_small_xai_summary.csv"

    if not os.path.exists(gradcam_summary_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {gradcam_summary_path}")

    df_gradcam = pd.read_csv(gradcam_summary_path).sort_values(by="regime_num")
    df_rollout = pd.read_csv(rollout_summary_path).sort_values(by="regime_num") if os.path.exists(rollout_summary_path) else None

    # Carrega métricas detalhadas de TP vs FN para Pointing Game do Grad-CAM
    tp_pointing = []
    fn_pointing = []
    for reg in [5, 20, 50, 100]:
        det_path = f"results/tables/vit_small_regime{reg}_gradcam_detailed_test.csv"
        if os.path.exists(det_path):
            det_df = pd.read_csv(det_path)
            tp_p = det_df[det_df["group"] == "TP"]["pointing_game"].mean()
            fn_p = det_df[det_df["group"] == "FN"]["pointing_game"].mean()
            tp_pointing.append(tp_p)
            fn_pointing.append(fn_p)
        else:
            tp_pointing.append(np.nan)
            fn_pointing.append(np.nan)

    df_gradcam["tp_pointing_game"] = tp_pointing
    df_gradcam["fn_pointing_game"] = fn_pointing

    return df_gradcam, df_rollout


def plot_figure1_clinical_evolution(df: pd.DataFrame, output_dir: str):
    """Figura 1: Evolução das Métricas Diagnósticas e Emergência de Falsos Negativos."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_labels = ["5%\n(740)", "20%\n(2.962)", "50%\n(7.407)", "100%\n(14.815)"]
    x = np.arange(len(x_labels))

    # 1. Taxa de Falsos Negativos (FNR %)
    ax = axes[0, 0]
    fnr_pct = df["fnr"] * 100
    bars = ax.bar(x, fnr_pct, color="#D9383A", width=0.55, edgecolor="black", alpha=0.9, zorder=3)
    ax.plot(x, fnr_pct, color="#8B0000", marker="o", linewidth=2.5, markersize=8, zorder=4)
    ax.set_title("A. Taxa de Falsos Negativos (FNR)")
    ax.set_ylabel("FNR (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 32)
    for i, bar in enumerate(bars):
        val = fnr_pct.iloc[i]
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", color="#8B0000")

    # 2. Contagem Absoluta de Falsos Negativos (Erros Clínicos)
    ax = axes[0, 1]
    fn_counts = df["fn_count"]
    bars_cnt = ax.bar(x, fn_counts, color="#E67E22", width=0.55, edgecolor="black", alpha=0.9, zorder=3)
    ax.plot(x, fn_counts, color="#B95C00", marker="s", linewidth=2.5, markersize=8, zorder=4)
    ax.set_title("B. Casos Absolutos de Falsos Negativos (Total Teste: 2.195)")
    ax.set_ylabel("Pacientes Doentes Declarados Sadios")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 670)
    for i, bar in enumerate(bars_cnt):
        val = fn_counts.iloc[i]
        diff_str = f" (-{((570 - val)/570)*100:.1f}%)" if i > 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, val + 20, f"{val}{diff_str}", ha="center", va="bottom", fontweight="bold")

    # 3. Sensibilidade vs Especificidade (Quebra do Viés Conservador)
    ax = axes[1, 0]
    ax.plot(x, df["sensitivity"] * 100, marker="o", color="#2E7D32", linewidth=2.5, markersize=8, label="Sensibilidade (Recall)")
    ax.plot(x, df["specificity"] * 100, marker="^", color="#1565C0", linewidth=2.5, markersize=8, linestyle="--", label="Especificidade")
    ax.set_title("C. Sensibilidade vs. Especificidade")
    ax.set_ylabel("Porcentagem (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(70, 103)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df["sensitivity"].iloc[i] * 100 - 3.5, f"{df['sensitivity'].iloc[i]*100:.1f}%", ha="center", color="#2E7D32", fontweight="bold")
        ax.text(x[i], df["specificity"].iloc[i] * 100 + 1.2, f"{df['specificity'].iloc[i]*100:.1f}%", ha="center", color="#1565C0", fontweight="bold")

    # 4. AUROC e Acurácia Balanceada
    ax = axes[1, 1]
    ax.plot(x, df["auroc"], marker="D", color="#6A1B9A", linewidth=2.5, markersize=8, label="AUROC")
    ax.plot(x, df["balanced_acc"], marker="v", color="#00838F", linewidth=2.5, markersize=8, linestyle="-.", label="Acurácia Balanceada")
    ax.set_title("D. AUROC e Acurácia Balanceada")
    ax.set_ylabel("Score [0, 1]")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.82, 1.01)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df["auroc"].iloc[i] + 0.008, f"{df['auroc'].iloc[i]:.4f}", ha="center", color="#6A1B9A", fontweight="bold")
        ax.text(x[i], df["balanced_acc"].iloc[i] - 0.015, f"{df['balanced_acc'].iloc[i]:.4f}", ha="center", color="#00838F", fontweight="bold")

    plt.suptitle("Dinâmica de Diagnóstico Clínico do Vision Transformer (ViT-Small) sob Escassez de Dados", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure1_clinical_evolution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura 1 salva em: {out_path}")


def plot_figure2_anatomical_reliability(df: pd.DataFrame, output_dir: str):
    """Figura 2: O Abismo Anatômico entre Acertos (TP) e Erros (FN) com Grad-CAM."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x_labels = ["5%\n(740)", "20%\n(2.962)", "50%\n(7.407)", "100%\n(14.815)"]
    x = np.arange(len(x_labels))
    width = 0.35

    # 1. Energia no Pulmão (EFL): TP vs FN
    ax = axes[0]
    b1 = ax.bar(x - width/2, df["tp_efl_mean"] * 100, width, label="Acertos (TP)", color="#2E7D32", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df["fn_efl_mean"] * 100, width, label="Falsos Negativos (FN)", color="#D9383A", edgecolor="black", alpha=0.9)
    ax.set_title("A. Energia Dentro do Pulmão (EFL %)\nAcertos (TP) vs Falsos Negativos (FN)")
    ax.set_ylabel("Fração de Energia no Pulmão (EFL %)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 52)
    ax.legend()
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#1B5E20")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#B71C1C")

    # 2. Vazamento Fora do Pulmão (Background Leakage): TP vs FN
    ax = axes[1]
    b1 = ax.bar(x - width/2, df["tp_leakage_mean"] * 100, width, label="Acertos (TP)", color="#81C784", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df["fn_leakage_mean"] * 100, width, label="Falsos Negativos (FN)", color="#E57373", edgecolor="black", alpha=0.9)
    ax.set_title("B. Vazamento Fora do Pulmão (Leakage %)\nDistração Anatômica por Grupo")
    ax.set_ylabel("Vazamento para o Fundo (1 - EFL %)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#B71C1C")

    # 3. Pointing Game (Mira na Lesão): TP vs FN
    ax = axes[2]
    b1 = ax.bar(x - width/2, df["tp_pointing_game"] * 100, width, label="Acertos (TP)", color="#1565C0", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df["fn_pointing_game"] * 100, width, label="Falsos Negativos (FN)", color="#C2185B", edgecolor="black", alpha=0.9)
    ax.set_title("C. Pointing Game (Hit Rate %)\nPico de Ativação Dentro do Pulmão")
    ax.set_ylabel("Pico de Ativação no Pulmão (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 75)
    ax.legend()
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#0D47A1")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color="#880E4F")

    plt.suptitle("Evidência de Confiabilidade Anatômica (Grad-CAM): O Abismo entre Acertos e Falsos Negativos", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure2_anatomical_reliability_gradcam.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura 2 salva em: {out_path}")


def plot_figure3_method_comparison(df_gradcam: pd.DataFrame, df_rollout: pd.DataFrame, output_dir: str):
    """Figura 3: Comparativo Metodológico entre Grad-CAM e Attention Rollout."""
    if df_rollout is None:
        print("[Aviso] Sumário do Rollout não encontrado. Pulando Figura 3.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_labels = ["5%\n(740)", "20%\n(2.962)", "50%\n(7.407)", "100%\n(14.815)"]
    x = np.arange(len(x_labels))

    # 1. Comparação de EFL Geral
    ax = axes[0]
    ax.plot(x, df_gradcam["overall_efl_mean"] * 100, marker="o", color="#D9383A", linewidth=2.5, markersize=8, label="Grad-CAM (Discriminativo com ReLU)")
    ax.plot(x, df_rollout["overall_efl_mean"] * 100, marker="s", color="#1565C0", linewidth=2.5, markersize=8, label="Attention Rollout (Atenção Global Multicamadas)")
    ax.set_title("A. Fração de Energia no Pulmão (EFL Geral %)\nGrad-CAM vs. Attention Rollout")
    ax.set_ylabel("EFL Geral (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(14, 34)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df_gradcam["overall_efl_mean"].iloc[i] * 100 + 0.8, f"{df_gradcam['overall_efl_mean'].iloc[i]*100:.1f}%", ha="center", color="#D9383A", fontweight="bold")
        ax.text(x[i], df_rollout["overall_efl_mean"].iloc[i] * 100 - 1.2, f"{df_rollout['overall_efl_mean'].iloc[i]*100:.1f}%", ha="center", color="#1565C0", fontweight="bold")

    # 2. Comparação de Entropia Espacial
    ax = axes[1]
    ax.plot(x, df_gradcam["overall_entropy_mean"], marker="o", color="#D9383A", linewidth=2.5, markersize=8, label="Grad-CAM (Mais focado/esparso)")
    ax.plot(x, df_rollout["overall_entropy_mean"], marker="s", color="#1565C0", linewidth=2.5, markersize=8, label="Attention Rollout (Mais difuso/névoa global)")
    ax.set_title("B. Dispersão Atencional (Entropia Espacial [0, 1])\nGrad-CAM vs. Attention Rollout")
    ax.set_ylabel("Entropia de Shannon Normalizada")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.91, 1.00)
    ax.legend(loc="lower right")
    for i in range(len(x)):
        ax.text(x[i], df_gradcam["overall_entropy_mean"].iloc[i] - 0.005, f"{df_gradcam['overall_entropy_mean'].iloc[i]:.3f}", ha="center", color="#D9383A", fontweight="bold")
        ax.text(x[i], df_rollout["overall_entropy_mean"].iloc[i] + 0.003, f"{df_rollout['overall_entropy_mean'].iloc[i]:.3f}", ha="center", color="#1565C0", fontweight="bold")

    plt.suptitle("Comparação Metodológica de XAI no Vision Transformer: Grad-CAM vs. Attention Rollout", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure3_method_comparison_rollout_vs_gradcam.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura 3 salva em: {out_path}")


def plot_figure4_vit_vs_resnet_comparison(df_vit: pd.DataFrame, output_dir: str):
    """Figura 4: O Confronto Científico Central: ViT-Small vs. ResNet-50 sob Escassez de Dados."""
    resnet_path = "results/tables/resnet50_gradcam_xai_summary.csv"
    if not os.path.exists(resnet_path):
        print("[Aviso] Sumário da ResNet-50 não encontrado. Pulando Figura 4.")
        return

    df_resnet = pd.read_csv(resnet_path).sort_values(by="regime_num")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    x_labels = ["5%\n(740)", "20%\n(2.962)", "50%\n(7.407)", "100%\n(14.815)"]
    x = np.arange(len(x_labels))
    width = 0.35

    # 1. Taxa de Falsos Negativos (FNR %)
    ax = axes[0, 0]
    b1 = ax.bar(x - width/2, df_resnet["fnr"] * 100, width, label="ResNet-50 (CNN)", color="#1976D2", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df_vit["fnr"] * 100, width, label="ViT-Small (Transformer)", color="#D32F2F", edgecolor="black", alpha=0.9)
    ax.plot(x - width/2, df_resnet["fnr"] * 100, color="#0D47A1", marker="o", linewidth=2)
    ax.plot(x + width/2, df_vit["fnr"] * 100, color="#B71C1C", marker="s", linewidth=2)
    ax.set_title("A. Taxa de Falsos Negativos (FNR %)\nViés Indutivo (CNN) vs Autoatenção (ViT)")
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

    # 2. Contagem Absoluta de Falsos Negativos
    ax = axes[0, 1]
    b1 = ax.bar(x - width/2, df_resnet["fn_count"], width, label="ResNet-50 (CNN)", color="#42A5F5", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df_vit["fn_count"], width, label="ViT-Small (Transformer)", color="#EF5350", edgecolor="black", alpha=0.9)
    ax.set_title("B. Casos Absolutos de Falsos Negativos (Total Teste: 2.195)")
    ax.set_ylabel("Pacientes Doentes Não Diagnosticados")
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

    # 3. AUROC e Sensibilidade (Recall)
    ax = axes[1, 0]
    ax.plot(x, df_resnet["auroc"], marker="o", color="#1976D2", linewidth=2.5, markersize=8, label="AUROC (ResNet-50)")
    ax.plot(x, df_vit["auroc"], marker="s", color="#D32F2F", linewidth=2.5, markersize=8, label="AUROC (ViT-Small)")
    ax.plot(x, df_resnet["sensitivity"], marker="^", color="#0D47A1", linewidth=2.0, markersize=7, linestyle="--", label="Sensibilidade (ResNet-50)")
    ax.plot(x, df_vit["sensitivity"], marker="v", color="#B71C1C", linewidth=2.0, markersize=7, linestyle="--", label="Sensibilidade (ViT-Small)")
    ax.set_title("C. AUROC e Sensibilidade Diagnóstica")
    ax.set_ylabel("Score [0, 1]")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower right")

    # 4. Energia Intrapulmonar Grad-CAM (EFL %)
    ax = axes[1, 1]
    b1 = ax.bar(x - width/2, df_resnet["overall_efl_mean"] * 100, width, label="EFL ResNet-50 (Grad-CAM)", color="#7E57C2", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + width/2, df_vit["overall_efl_mean"] * 100, width, label="EFL ViT-Small (Grad-CAM)", color="#26A69A", edgecolor="black", alpha=0.9)
    ax.set_title("D. Fração de Energia no Pulmão (EFL %)\nConfiabilidade Anatômica dos Gradientes")
    ax.set_ylabel("EFL Geral (%)")
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

    plt.suptitle("O Confronto Científico: ResNet-50 vs. ViT-Small sob Escassez de Dados em Radiografias de Tórax", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure4_vit_vs_resnet_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura 4 salva em: {out_path}")


def main():
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    df_gradcam, df_rollout = load_data()

    plot_figure1_clinical_evolution(df_gradcam, output_dir)
    plot_figure2_anatomical_reliability(df_gradcam, output_dir)
    plot_figure3_method_comparison(df_gradcam, df_rollout, output_dir)
    plot_figure4_vit_vs_resnet_comparison(df_gradcam, output_dir)
    print("\nTodas as figuras científicas foram geradas com sucesso em alta resolução (300 DPI)!")


if __name__ == "__main__":
    main()
