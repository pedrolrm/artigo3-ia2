# Análise da Emergência de Falsos Negativos e Confiabilidade Anatômica em CNNs e Vision Transformers sob Escassez de Dados em Radiografias de Tórax

Este repositório contém o código e os pipelines experimentais para o estudo comparativo entre Redes Neurais Convolucionais (**ResNet-50**) e Vision Transformers (**ViT-Small**) sob diferentes regimes de volume de dados em radiografias de tórax, com foco na emergência e explicação de **Falsos Negativos (FN)** via **Grad-CAM**.

---

## Estrutura do Repositório

```text
artigo3-ia2/
├── .gitignore                   # Ignora dados brutos, checkpoints e artefatos pesados
├── AGENTS.md                    # Especificação técnica, científica e metodológica completa
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências do projeto
├── data/
│   ├── raw/                     # Imagens e máscaras (ignorado no git)
│   └── manifests/               # Manifestos CSV estratificados (train, val, test)
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Carregador de raio-X e máscaras com pareamento
│   ├── models/                  # Subpacote de arquiteturas
│   │   ├── __init__.py          # Fábrica de modelos unificada
│   │   ├── resnet.py            # Instanciação da ResNet-50 via timm e camada Grad-CAM
│   │   └── vit.py               # Instanciação do ViT-Small via timm e reshape Grad-CAM
│   ├── train.py                 # Loop de treinamento com validação e early stopping
│   ├── evaluate.py              # Cálculo de métricas diagnósticas (AUROC, FNR, etc.)
│   ├── xai.py                   # Implementação do Grad-CAM (CNN e ViT com reshape)
│   └── metrics_xai.py           # Cálculo de Entropia Espacial e EFL com máscaras
├── scripts/
│   ├── 01_prepare_data.py       # Extração do ZIP e geração dos manifestos CSV
│   ├── 02_run_experiments.py    # Execução sequencial dos treinos para todos os regimes
│   └── 03_analyze_fn_xai.py     # Extração de XAI no Test Set e compilação dos dados
└── results/
    ├── checkpoints/             # Melhores pesos salvos (ignorado no git)
    ├── logs/                    # Históricos de treinamento (ignorado no git)
    ├── tables/                  # CSVs compilados com métricas (ignorado no git)
    └── figures/                 # Gráficos e painéis qualitativos (ignorado no git)
```

---

## Diretrizes e Protocolo Experimental

Para todos os detalhes metodológicos, pareamento de parâmetros, formulação das métricas (Entropia Espacial, EFL, Background Leakage) e divisão estratificada dos regimes, consulte o documento:
* [AGENTS.md](AGENTS.md)
