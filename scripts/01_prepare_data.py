"""01_prepare_data.py
Pipeline de preparação do COVID-19 Radiography Database.
Gera os manifestos CSV determinísticos e aninhados para treino, validação e teste.
Mapeamento binário:
  - Classe 0: Saudável (Normal)
  - Classe 1: Não Saudável (COVID, Lung_Opacity, Viral Pneumonia)
"""

import os
import argparse
import zipfile
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split


def locate_or_extract_dataset(
    data_dir: Optional[str] = None,
    zip_path: Optional[str] = None,
    raw_dir: str = "data/raw"
) -> str:
    """
    Localiza o diretório do dataset ou extrai automaticamente do arquivo ZIP
    conforme as diretrizes do AGENTS.md.
    """
    candidates = []
    if data_dir:
        candidates.append(data_dir)
    candidates.extend([
        os.path.join(raw_dir, "COVID-19_Radiography_Dataset"),
        raw_dir,
        r"C:\Users\pluca\Downloads\COVID-19_Radiography_Dataset",
        r"C:\Users\jonat\Desktop\ARTIGO 3\DATASET\COVID-19_Radiography_Dataset",
    ])

    for cand in candidates:
        if os.path.exists(cand) and os.path.exists(os.path.join(cand, "Normal", "images")):
            print(f"Dataset encontrado em: {cand}")
            return cand

    # Se não encontrado nas pastas, tenta extrair do arquivo ZIP
    zip_candidates = []
    if zip_path:
        zip_candidates.append(zip_path)
    zip_candidates.extend([
        r"C:\Users\pluca\Downloads\COVID-19_Radiography_Dataset.zip",
        os.path.join(raw_dir, "COVID-19_Radiography_Dataset.zip"),
        "COVID-19_Radiography_Dataset.zip",
    ])

    found_zip = None
    for zc in zip_candidates:
        if os.path.exists(zc):
            found_zip = zc
            break

    if found_zip:
        print(f"Arquivo ZIP encontrado: {found_zip}")
        print(f"Extraindo para {raw_dir} (pode levar alguns instantes)...")
        os.makedirs(raw_dir, exist_ok=True)
        with zipfile.ZipFile(found_zip, "r") as zf:
            zf.extractall(raw_dir)
        print("Extração concluída!")

        for cand in candidates:
            if os.path.exists(cand) and os.path.exists(os.path.join(cand, "Normal", "images")):
                return cand

    raise FileNotFoundError(
        "Diretório do dataset ou arquivo ZIP não encontrado. "
        "Especifique via --data_dir ou coloque o arquivo ZIP em data/raw/ ou Downloads."
    )


def build_manifest(data_dir: str):
    categories = {
        "Normal": 0,
        "COVID": 1,
        "Lung_Opacity": 1,
        "Viral Pneumonia": 1
    }

    records = []
    print(f"Buscando imagens em: {data_dir}")

    for cat_name, label in categories.items():
        img_dir = os.path.join(data_dir, cat_name, "images")
        mask_dir = os.path.join(data_dir, cat_name, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {img_dir} ou {mask_dir}")

        filenames = sorted(os.listdir(img_dir))
        matched = 0

        for fname in filenames:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)

            if not os.path.exists(mask_path):
                print(f"[AVISO] Máscara ausente para: {img_path}")
                continue

            records.append({
                "image_path": os.path.abspath(img_path),
                "mask_path": os.path.abspath(mask_path),
                "category": cat_name,
                "label": label,
                "label_name": "Saudável" if label == 0 else "Não Saudável"
            })
            matched += 1

        print(f"  -> {cat_name}: {matched} pares de imagem/máscara encontrados (Classe {label}).")

    df = pd.DataFrame(records)
    print(f"\nTotal consolidado: {len(df)} amostras.")
    print("Distribuição binária:")
    print(df["label_name"].value_counts(normalize=False))
    print(df["label_name"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    return df


def split_and_save_manifests(df: pd.DataFrame, output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Test set fixo (20%) estratificado por categoria original
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["category"]
    )

    # 2. Val set fixo (10% do total = 10/80 = 0.125 do train_val_df)
    train_100_df, val_df = train_test_split(
        train_val_df,
        test_size=0.125,
        random_state=seed,
        stratify=train_val_df["category"]
    )

    # 3. Regimes aninhados de treino:
    # train_50 (50% de train_100)
    train_50_df, _ = train_test_split(
        train_100_df,
        train_size=0.50,
        random_state=seed,
        stratify=train_100_df["category"]
    )

    # train_20 (20% de train_100 = 40% de train_50)
    train_20_df, _ = train_test_split(
        train_50_df,
        train_size=0.40,
        random_state=seed,
        stratify=train_50_df["category"]
    )

    # train_5 (5% de train_100 = 25% de train_20)
    train_5_df, _ = train_test_split(
        train_20_df,
        train_size=0.25,
        random_state=seed,
        stratify=train_20_df["category"]
    )

    # Salvando CSVs
    splits = {
        "test.csv": test_df,
        "val.csv": val_df,
        "train_100.csv": train_100_df,
        "train_50.csv": train_50_df,
        "train_20.csv": train_20_df,
        "train_5.csv": train_5_df,
    }

    print("\n--- Manifestos Gerados ---")
    for fname, split_df in splits.items():
        out_path = os.path.join(output_dir, fname)
        split_df.to_csv(out_path, index=False)
        pos = (split_df["label"] == 1).sum()
        neg = (split_df["label"] == 0).sum()
        print(f"Salvo: {fname:15s} | Total: {len(split_df):5d} | Saudável: {neg:5d} | Não Saudável: {pos:5d}")


def main():
    parser = argparse.ArgumentParser(description="Prepara manifestos do COVID-19 Radiography Dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Caminho para o diretório raiz descompactado do dataset"
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="Caminho alternativo para o arquivo ZIP do dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"data\manifests",
        help="Diretório de saída para os manifestos CSV"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para estratificação"
    )
    args = parser.parse_args()

    dataset_path = locate_or_extract_dataset(data_dir=args.data_dir, zip_path=args.zip_path)
    df = build_manifest(dataset_path)
    split_and_save_manifests(df, args.output_dir, seed=args.seed)
    print("\nPreparação de dados concluída com sucesso!")


if __name__ == "__main__":
    main()
