"""scripts/02_run_experiments.py
Orquestrador sequencial dos experimentos de treino por regime de dados.
Otimizado para execução acelerada na NVIDIA GeForce RTX 5060 (8GB).
"""

import os
import sys

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Executa os experimentos de treino por regime")
    parser.add_argument(
        "--architecture",
        type=str,
        default="vit_small",
        choices=["vit_small", "resnet50", "all"],
        help="Arquitetura a treinar. Usuário responsável pelo ViT: 'vit_small'"
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["5", "20", "50", "100"],
        help="Lista de regimes percentuais a executar (ex: 5 20 50 100)"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Tamanho do lote otimizado para os 8GB VRAM da RTX 5060")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Desativa bfloat16 mixed precision")
    args = parser.parse_args()

    architectures = ["vit_small"] if args.architecture == "vit_small" else (
        ["resnet50"] if args.architecture == "resnet50" else ["vit_small", "resnet50"]
    )

    print("================================================================")
    print(" INICIANDO PIPELINE DE EXPERIMENTOS DE TREINAMENTO")
    print(f" Arquiteturas: {architectures}")
    print(f" Regimes: {args.regimes}")
    print(f" Batch Size: {args.batch_size} | Workers: {args.num_workers}")
    print(f" Aceleração AMP (bfloat16): {not args.no_amp}")
    print(f" Épocas máx: {args.num_epochs} | Paciência Early Stopping: {args.patience}")
    print("================================================================\n")

    summary_records = []

    for arch in architectures:
        for reg in args.regimes:
            print(f"\n>>> Rodando {arch} no regime {reg}% de treino...")
            res = train_model(
                architecture=arch,
                regime=reg,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
                num_workers=args.num_workers,
                use_amp=(not args.no_amp)
            )
            summary_records.append({
                "architecture": arch,
                "regime": reg,
                "best_epoch": res["best_epoch"],
                "best_val_auroc": res["best_val_auroc"],
                "checkpoint": res["checkpoint_path"]
            })

    print("\n================================================================")
    print(" TODOS OS TREINAMENTOS PROGRAMADOS FORAM CONCLUÍDOS!")
    print("================================================================")
    for rec in summary_records:
        print(f"[{rec['architecture']}] Regime {rec['regime']:>3}% -> Val AUROC: {rec['best_val_auroc']:.4f} (Época {rec['best_epoch']})")


if __name__ == "__main__":
    main()
