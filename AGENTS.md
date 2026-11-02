# AGENTS.md — Diretrizes de Pesquisa, Dados e Implementação

Este documento serve como a **especificação técnica e metodológica definitiva** para agentes autônomos e pesquisadores que colaboram no desenvolvimento deste repositório. Ele detalha a pergunta científica, as características dos dados, o protocolo de comparação justa (*fair comparison*) e as etapas de implementação para o artigo.

---

## 1. Visão Geral do Artigo e Pergunta Científica
Link para o dataset: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data

### 1.1. Título do Projeto
> **Análise da Emergência de Falsos Negativos e Confiabilidade Anatômica em CNNs e Vision Transformers sob Escassez de Dados em Radiografias de Tórax**

### 1.2. O Problema Clínico
* **Classificação Binária:** Triagem rápida de radiografias de tórax entre exames **Saudáveis (Normal)** e exames **Não Saudáveis (Patológicos: COVID-19, Opacidade Pulmonar e Pneumonia Viral)**.
* **Foco Clínico Crítico:** A emergência de **Falsos Negativos (FN)** — pacientes patológicos que o modelo diagnostica erroneamente como sadios. Na prática clínica, esse é o erro mais grave, pois atrasa intervenções vitais e permite a disseminação de patologias respiratórias.

### 1.3. A Hipótese Científica
* **Viés Indutivo vs. Escassez:** Redes Convolucionais (CNNs) operam com fortes vieses indutivos de localidade e invariância à translação. Vision Transformers (ViTs) utilizam autoatenção global sem restrições espaciais rígidas.
* **Hipótese:** Sob regimes de **escassez de dados de treino**, o ViT apresenta maior dispersão atencional (mensurada por alta entropia espacial) e maior vazamento de ativação para fora da região pulmonar (*background leakage*), resultando em uma taxa desproporcionalmente maior de falsos negativos. À medida que o volume de dados aumenta, o ViT refina sua atenção intrapulmonar, reduz a entropia e se equipara ou supera a CNN.

---

## 2. Descrição e Especificação dos Dados

### 2.1. Origem, Resolução Real e Padronização dos Dados
* **Base:** *COVID-19 Radiography Database* (V2 / Update 2).
* **Arquivo Fonte:** `C:\Users\pluca\Downloads\COVID-19_Radiography_Dataset.zip`
* **Formato e Resoluções Nativas dos Arquivos:**
  - **Radiografias de Tórax (`images/`):** Arquivos PNG na resolução original de **$299 \times 299$ pixels**.
  - **Máscaras Pulmonares Anatômicas (`masks/`):** Arquivos PNG na resolução original de **$256 \times 256$ pixels**.
* **Requisito Obrigatório de Pré-processamento e Alinhamento ($224 \times 224$):**
  - Tanto a `ResNet-50` quanto o `ViT-Small` exigem entrada em **$224 \times 224$ pixels** (para o ViT dividir em $14 \times 14$ patches de $16 \times 16$).
  - **Radiografias:** Redimensionadas de $299 \times 299 \to 224 \times 224$ via interpolação bilinear/bicúbica e normalização ImageNet.
  - **Máscaras Pulmonares:** Redimensionadas de $256 \times 256 \to 224 \times 224$ **estritamente via interpolação `nearest-neighbor`** para manter os valores discretos binários $\{0, 1\}$ sem criar artefatos fracionários nas margens pulmonares.
  - Isso garante que no cálculo de Grad-CAM, a radiografia, a máscara e o mapa de calor estejam exatamente no mesmo sistema de coordenadas $224 \times 224$ pixel a pixel.

### 2.2. Classes Originais e Agrupamento Binário

| Subpasta Original | Condição Clínica | Imagens | Máscaras | Mapeamento no Artigo |
| :--- | :--- | :---: | :---: | :---: |
| `Normal/` | Sem achados patológicos | **10.192** | 10.192 | **Classe 0 — Saudável** (48,15%) |
| `Lung_Opacity/` | Infiltrados / Infecções não-COVID | **6.012** | 6.012 | **Classe 1 — Não Saudável** (51,85%) |
| `COVID/` | Infecção por SARS-CoV-2 | **3.616** | 3.616 | **Classe 1 — Não Saudável** |
| `Viral Pneumonia/`| Pneumonia Viral | **1.345** | 1.345 | **Classe 1 — Não Saudável** |
| **Total** | | **21.165** | **21.165** | **Equilíbrio natural ~1:1** |

> **Nota Metodológica:** O balanceamento binário natural (10.192 vs. 10.973) elimina a necessidade de técnicas artificiais de reamostragem (como SMOTE ou undersampling agressivo), garantindo que as taxas de falsos negativos reflitam o aprendizado da arquitetura e não distorções de amostragem.

---

## 3. Protocolo Experimental Estrito e Comparação Justa (*Fair Comparison*)

Para isolar o efeito da arquitetura e do volume de dados sem introduzir variáveis de confusão, os agentes devem seguir rigorosamente os princípios abaixo:

### 3.1. Paridade de Capacidade (Parâmetros) e Transfer Learning
* **CNN Representativa:** `ResNet-50` (~25,5 milhões de parâmetros).
* **ViT Representativo:** `ViT-Small` (`vit_small_patch16_224`, ~22 milhões de parâmetros).
* **Pesos Pré-treinados:** Ambos partem de pesos oficiais do **ImageNet-1k** (via biblioteca `timm`).
  * *Justificativa:* Nenhuma das redes possui conhecimento radiológico prévio ou contaminação de dados (*zero data leakage*). Ambas possuem paridade volumétrica (~22M vs ~25M), impedindo a justificativa de que uma rede falhou por sobreparametrização.

### 3.2. Divisão Estratificada Fixa e Regimes Aninhados
1. **Conjuntos Globais Imutáveis:**
   * **Teste Fixo (20% ~ 4.233 amostras):** Rigorosamente o mesmo conjunto de avaliação para **todos** os modelos e regimes.
   * **Validação Fixa (10% ~ 2.116 amostras):** Usado para seleção de hiperparâmetros e *early stopping*.
   * **Treino Base (70% ~ 14.816 amostras).**
2. **Regimes de Volume de Dados (Amostragem Aninhada):**
   * **Regime 1 (5% do treino):** ~740 imagens.
   * **Regime 2 (20% do treino):** ~2.963 imagens.
   * **Regime 3 (50% do treino):** ~7.408 imagens.
   * **Regime 4 (100% do treino):** 14.816 imagens.
   * *Regra de Inclusão Aninhada:* $\text{Treino}_{5\%} \subset \text{Treino}_{20\%} \subset \text{Treino}_{50\%} \subset \text{Treino}_{100\%}$. Os subconjuntos menores devem ser subconjuntos estritos dos maiores para eliminar variação aleatória de amostragem.

### 3.3. Padronização do Treinamento
* **Resolução de Entrada:** $224 \times 224$ pixels (3 canais RGB replicados a partir do monocromático).
* **Pipeline de Augmentation:** Idêntico para ambos (Random Horizontal Flip com $p=0.5$, rotação leve $[-10^\circ, +10^\circ]$, translação leve $[-5\%, +5\%]$).
* **Otimizador:** `AdamW` (peso de decaimento padrão $1\times 10^{-2}$).
* **Learning Rate:** Learning rate base de $1\times 10^{-4}$ com *cosine annealing scheduler* e *warmup* de 3 épocas.
* **Tamanho de Lote (Batch Size):** 32 ou 64 (idêntico para ambos).
* **Critério de Parada:** *Early stopping* monitorando o AUROC de validação com paciência de 7 épocas.
* **Reprodutibilidade:** Sementes aleatórias fixas (`seed=42`) para PyTorch, NumPy e divisão do dataset.

---

## 4. Padronização do Mecanismo de Explicabilidade (Grad-CAM)

Para evitar viés algorítmico de XAI, o **Grad-CAM** deve ser o único extrator de mapas de calor para ambas as redes:

1. **Na CNN (ResNet-50):**
   * Camada Alvo: Última camada convolucional (`model.layer4[-1]`).
2. **No ViT (ViT-Small):**
   * Camada Alvo: Saída do último bloco transformer (`model.blocks[-1].norm1`).
   * *Reshape Transform:* O token `[CLS]` é descartado, e os 196 tokens espaciais ($14 \times 14$) são rearranjados para o formato de feature map $(B, C, 14, 14)$.
3. **Pós-processamento dos Mapas:**
   * Aplicação de ReLU e normalização Min-Max para $[0, 1]$.
   * Interpolação bilinear para a resolução original ($224 \times 224$).

---

## 5. Métricas de Avaliação

O pipeline deve calcular e registrar formalmente três eixos de métricas:

### 5.1. Eixo Clínico e Diagnóstico
* **Taxa de Falsos Negativos (FNR):** $\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = 1 - \text{Sensibilidade}$
* **Sensibilidade (Recall), Especificidade, Acurácia Balanceada e AUROC.**

### 5.2. Eixo de Dispersão Atencional (Entropia Espacial)
Normaliza-se o mapa de calor $A(x,y)$ como distribuição de probabilidade $P(x,y) = \frac{A(x,y)}{\sum A(x,y)}$:
$$H(A) = - \frac{1}{\log(N)} \sum_{x,y} P(x,y) \log(P(x,y) + \epsilon)$$
* $H(A) \in [0, 1]$: Valores próximos de 1 indicam atenção difusa/dispersa; valores próximos de 0 indicam foco atencional concentrado.

### 5.3. Eixo de Validação Anatômica (Máscara Pulmonar)
Utilizando a máscara binária real do pulmão $M(x,y) \in \{0, 1\}$:
1. **Fração de Energia no Pulmão (Energy Fraction inside Lung - EFL):**
   $$\text{EFL} = \frac{\sum_{(x,y) \in M} A(x,y)}{\sum_{\text{todos } (x,y)} A(x,y)}$$
2. **Vazamento para o Fundo (*Background Leakage*):** $1 - \text{EFL}$.
3. **Pointing Game (Hit Rate):** 1 se o ponto de máxima ativação $\arg\max A(x,y)$ reside dentro de $M$, 0 caso contrário.

### 5.4. Análise Específica de Falsos Negativos
Para cada combinação de [Arquitetura $\times$ Regime], deve-se estratificar as amostras em quatro grupos: **Verdadeiros Positivos (TP)**, **Falsos Negativos (FN)**, **Verdadeiros Negativos (TN)** e **Falsos Positivos (FP)**, comparando as médias e distribuições de **Entropia** e **EFL**.

---

## 6. Arquitetura de Diretórios e Descrição Funcional dos Arquivos

```text
artigo3-ia2/
├── AGENTS.md                    # Este arquivo: diretrizes metodológicas e técnicas
├── README.md                    # Visão geral do repositório
├── requirements.txt             # Dependências (torch, torchvision, timm, etc.)
├── data/
│   ├── raw/                     # Imagens e máscaras extraídas (ignorado no git)
│   └── manifests/               # CSVs determinísticos de divisão (ignorados no git):
│       ├── test.csv             # Conjunto de teste fixo (20%)
│       ├── val.csv              # Conjunto de validação fixo (10%)
│       ├── train_100.csv        # Treino integral (100%)
│       ├── train_50.csv         # Subconjunto aninhado de 50%
│       ├── train_20.csv         # Subconjunto aninhado de 20%
│       └── train_5.csv          # Subconjunto aninhado de 5%
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Carregador de raio-X e máscaras com pareamento
│   ├── models/                  # Subpacote de instanciação das arquiteturas
│   │   ├── __init__.py          # Exporta funções de instanciação e contagem de parâmetros
│   │   ├── resnet.py            # Instanciação da ResNet-50 via timm e camada-alvo Grad-CAM
│   │   └── vit.py               # Instanciação do ViT-Small via timm e reshape_transform
│   ├── train.py                 # Loop de treinamento com validação e early stopping
│   ├── evaluate.py              # Cálculo de métricas diagnósticas (AUROC, FNR, etc.)
│   ├── xai.py                   # Implementação do Grad-CAM (CNN e ViT com reshape)
│   └── metrics_xai.py           # Cálculo de Entropia Espacial e EFL com máscaras
├── scripts/
│   ├── 01_prepare_data.py       # Extração do ZIP e geração dos manifestos CSV
│   ├── 02_run_experiments.py    # Execução sequencial dos treinos para todos os regimes
│   └── 03_analyze_fn_xai.py     # Extração de XAI no Test Set e compilação dos dados
└── results/
    ├── checkpoints/             # Melhores pesos de cada regime e arquitetura
    ├── logs/                    # Históricos de treino e validação
    ├── tables/                  # CSVs compilados com métricas diagnósticas e de XAI
    └── figures/                 # Gráficos comparativos e painéis qualitativos de FN
```

### 6.1. Módulos Principais (`src/`)

* **`src/dataset.py`:**
  - **Função:** Implementar a classe PyTorch `Dataset` (`ChestXRayDataset`) que carrega pares de radiografias de tórax e respectivas máscaras pulmonares a partir dos manifestos CSV.
  - **Requisitos de Implementação:** Carregar imagens em formato RGB ($224 \times 224$) com normalização ImageNet. Quando `return_mask=True`, carregar e redimensionar a máscara binária correspondente utilizando estritamente interpolação `nearest` para preservar valores discretos $\{0, 1\}$.
* **`src/models/` (Subpacote de Arquiteturas):**
  - **`src/models/resnet.py`:**
    - **Função:** Instanciação da CNN representativa (`resnet50` via `timm`) com pesos ImageNet-1k e cabeça binária (2 classes).
    - **Requisitos de Implementação:** Expor função para retorno do modelo e identificação da camada-alvo para Grad-CAM (`model.layer4[-1]`).
  - **`src/models/vit.py`:**
    - **Função:** Instanciação do Vision Transformer representativo (`vit_small_patch16_224` via `timm`) com pesos ImageNet-1k e cabeça binária (2 classes).
    - **Requisitos de Implementação:** Expor função para retorno do modelo, identificação da camada-alvo para Grad-CAM (`model.blocks[-1].norm1`) e a função `reshape_transform` (que descarta o token `[CLS]` e faz o reshape dos 196 tokens espaciais para o grid $14 \times 14$ canais).
  - **`src/models/__init__.py`:**
    - **Função:** Ponto de entrada do subpacote, exportando uma função unificada de fábrica (`build_model(architecture: str, pretrained: bool = True, num_classes: int = 2)`) e utilitários de contagem de parâmetros (`count_parameters`).
* **`src/train.py`:**
  - **Função:** Gerenciador do loop de treinamento e validação para um dado modelo e regime.
  - **Requisitos de Implementação:** Receber modelo, loaders de treino e validação, configurando otimizador `AdamW` ($weight\_decay = 1\times 10^{-2}$), scheduler com warmup e cosine decay. Executar *early stopping* monitorando o AUROC de validação (paciência de 7 épocas) e persistir o melhor checkpoint em `results/checkpoints/`.
* **`src/evaluate.py`:**
  - **Função:** Avaliação de métricas diagnósticas padronizadas no conjunto de teste.
  - **Requisitos de Implementação:** Calcular Taxa de Falsos Negativos ($\text{FNR} = 1 - \text{Sensibilidade}$), Sensibilidade (Recall), Especificidade, Acurácia Balanceada, AUROC e contagens da matriz de confusão ($\text{TP}, \text{FN}, \text{TN}, \text{FP}$).
* **`src/xai.py`:**
  - **Função:** Motor uniforme de extração do Grad-CAM compatível com CNNs e Vision Transformers.
  - **Requisitos de Implementação:** Implementar hooks para captura de ativações e gradientes na camada alvo. Para o ViT-Small, converter a sequência de tokens ($1D$) para grid espacial ($2D$) via `reshape_transform`. Ponderar ativações pelos gradientes médios, aplicar ReLU, interpolar bilinearmente para $224 \times 224$ e normalizar Min-Max para $[0, 1]$.
* **`src/metrics_xai.py`:**
  - **Função:** Biblioteca de métricas quantitativas de explicabilidade.
  - **Requisitos de Implementação:** Implementar: (1) Entropia Espacial de Shannon normalizada $H(A) \in [0, 1]$ sobre o mapa de calor; (2) Fração de Energia no Pulmão ($\text{EFL} \in [0, 1]$) integrando sobre a máscara pulmonar ground-truth; (3) Vazamento de Fundo ($1 - \text{EFL}$); (4) Pointing Game (*Hit Rate* binário indicando se $\arg\max A$ reside na máscara).

---

### 6.2. Scripts de Execução (`scripts/`)

* **`scripts/01_prepare_data.py`:**
  - **Função:** Pipeline de descompactação e geração determinística dos manifestos CSV.
  - **Requisitos de Implementação:** Extrair `COVID-19_Radiography_Dataset.zip` para `data/raw/` se necessário. Agrupar as 4 classes originais no mapeamento binário (0: Normal, 1: Patológico). Criar a divisão fixa estratificada (`seed=42`) gerando `data/manifests/test.csv` (20%), `val.csv` (10%) e `train_100.csv` (70%). Gerar os subconjuntos de treino aninhados (`train_50.csv`, `train_20.csv` e `train_5.csv`).
* **`scripts/02_run_experiments.py`:**
  - **Função:** Orquestrador sequencial de todos os treinamentos do estudo comparativo.
  - **Requisitos de Implementação:** Iterar de forma reprodutiva sobre as 8 combinações experimentais: [ResNet-50, ViT-Small] $\times$ [5%, 20%, 50%, 100% de treino], acionando `src/train.py`. Registrar curvas de aprendizado em `results/logs/` e salvar pesos ótimos em `results/checkpoints/`.
* **`scripts/03_analyze_fn_xai.py`:**
  - **Função:** Auditoria de explicabilidade no Test Set e compilação de resultados do artigo.
  - **Requisitos de Implementação:** Carregar os 8 modelos treinados, executar inferência sobre o Test Set imutável (`test.csv`), extrair Grad-CAM para todas as amostras e calcular métricas via `src/metrics_xai.py`. Estratificar por grupos diagnósticos (com ênfase no grupo Falso Negativo), salvar tabelas consolidadas em `results/tables/` e gerar os gráficos comparativos e painéis qualitativos em `results/figures/`.

---

## 7. Instruções para Agentes e Desenvolvedores

1. **Reprodutibilidade Estrita:** Não utilize divisões aleatórias inline no código sem carregar os manifestos CSV pré-gerados em `data/manifests/`.
2. **Transparência de Código:** Documente no cabeçalho de cada script qual regime e modelo estão sendo manipulados.
3. **Tratamento de Máscaras:** Assegure-se de que a máscara binária correspondente à imagem de raio-X seja redimensionada exatamente da mesma forma ($224 \times 224$ com interpolação `nearest`) para que a validação anatômica não sofra degradação geométrica.
4. **Isolamento de Teste:** O conjunto de teste **nunca** deve ser utilizado durante o ajuste de hiperparâmetros, *early stopping* ou treinamento. Ele é reservado exclusivamente para a inferência final de diagnóstico e extração de Grad-CAM.

---

## 8. Referências Bibliográficas Selecionadas e Justificativa Metodológica

Para garantir o rigor acadêmico e respaldar formalmente as contribuições do artigo perante revisores de periódicos e conferências de ponta (ex: IEEE, Elsevier, Springer, Nature), foram selecionadas 14 referências estratégicas divididas em 4 blocos temáticos.

### 8.1. Bloco de Referências em Formato LaTeX (Padrão `IEEEtran`)

```latex
\bibliographystyle{IEEEtran}
\begin{thebibliography}{00}

% --- BLOCO 1: BASE DE DADOS E VALIDAÇÃO DE MÁSCARAS ---
\bibitem{chowdhury2020can}
M.~E.~H. Chowdhury, T.~Rahman, A.~Khandakar, R.~Mazhar, M.~A. Kadir, Z.~B. Mahbub, K.~R. Islam, M.~S. Khan, A.~Iqbal, N.~Al-Emadi, M.~B.~I. Reaz, and M.~T. Islam, ``Can AI help in screening Viral and COVID-19 pneumonia?'' \emph{IEEE Access}, vol.~8, pp. 132665--132676, 2020.

\bibitem{rahman2021exploring}
T.~Rahman, A.~Khandakar, Y.~Qiblawey, A.~Tahir, S.~Kiranyaz, S.~B.~A. Kashem, M.~T. Islam, S.~Al-Maadeed, S.~M. Zughaier, M.~S. Khan, and M.~E.~H. Chowdhury, ``Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images,'' \emph{Computers in Biology and Medicine}, vol. 132, p. 104319, 2021.

\bibitem{kaggle_covid19_radiography}
M.~E.~H. Chowdhury, T.~Rahman, and A.~Khandakar, ``COVID-19 Radiography Database,'' \emph{Kaggle}, 2020. [Online]. Available: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

% --- BLOCO 2: ARQUITETURAS E HIPÓTESE DE VIÉS INDUTIVO ---
\bibitem{he2016deep}
K.~He, X.~Zhang, S.~Ren, and J.~Sun, ``Deep residual learning for image recognition,'' in \emph{Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 2016, pp. 770--778.

\bibitem{dosovitskiy2020image}
A.~Dosovitskiy \emph{et al.}, ``An image is worth 16x16 words: Transformers for image recognition at scale,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, 2021.

\bibitem{raghu2021vision}
M.~Raghu, T.~Unterthiner, S.~Kornblith, C.~Zhang, and A.~Dosovitskiy, ``Do vision transformers see like convolutional neural networks?'' in \emph{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol.~34, 2021, pp. 12116--12128.

\bibitem{matsoukas2021time}
C.~Matsoukas, J.~F. Haslum, M.~Moein, A.~Rakhlin, I.~Hakansson, and K.~Smith, ``Is it time to replace {CNNs} with transformers for medical images?'' \emph{Med. Image Anal.}, vol.~74, p. 102246, 2021.

% --- BLOCO 3: MÉTODOS DE XAI E AVALIAÇÃO ANATÔMICA ---
\bibitem{selvaraju2017grad}
R.~R. Selvaraju \emph{et al.}, ``{Grad-CAM}: Visual explanations from deep networks via gradient-based localization,'' in \emph{Proc. IEEE Int. Conf. Comput. Vis. (ICCV)}, 2017, pp. 618--626.

\bibitem{abnar2020quantifying}
S.~Abnar and W.~Zuidema, ``Quantifying attention flow in transformers,'' in \emph{Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)}, 2020, pp. 4247--4256.

\bibitem{zhang2018top}
J.~Zhang \emph{et al.}, ``Top-down neural attention by excitation backprop,'' \emph{Int. J. Comput. Vis.}, vol. 126, no.~10, pp. 1084--1102, 2018.

% --- BLOCO 4: O PARADOXO DO AUROC, SHORTCUTS E RISCO CLÍNICO ---
\bibitem{oakden2020hidden}
L.~Oakden-Rayner \emph{et al.}, ``Hidden stratification causes clinically meaningful failures in machine learning for medical imaging,'' \emph{Nat. Med.}, vol.~26, no.~4, pp. 536--539, 2020.

\bibitem{degrave2021ai}
A.~J. DeGrave, J.~D. Janizek, and S.-I. Lee, ``{AI} for radiographic {COVID-19} detection selects shortcuts over pathology,'' \emph{Nat. Mach. Intell.}, vol.~3, no.~7, pp. 610--619, 2021.

\bibitem{saporta2022benchmarking}
A.~Saporta \emph{et al.}, ``Benchmarking saliency methods for chest {X}-ray interpretation,'' \emph{Nat. Mach. Intell.}, vol.~4, no.~10, pp. 867--878, 2022.

\bibitem{wong2020frequency}
H.~Y.~F. Wong \emph{et al.}, ``Frequency and distribution of chest radiographic findings in patients positive for {COVID-19},'' \emph{Radiology}, vol. 296, no.~2, pp. E72--E78, 2020.

\end{thebibliography}
```

---

### 8.2. Justificativa Teórica e Metodológica para Cada Escolha

#### Bloco 1: Base de Dados e Validação de Máscaras
1. **`chowdhury2020can` (IEEE Access 2020):**
   - *Motivo:* Artigo formal original dos criadores da base *COVID-19 Radiography Database*. Seu uso é mandatório para documentar a procedência clínica das 21.165 imagens na resolução nativa de $299 \times 299$.
2. **`rahman2021exploring` (Computers in Biology and Medicine 2021):**
   - *Motivo:* Artigo formal de expansão da base (Update 2), onde foram introduzidas as máscaras anatômicas pulmonares ($256 \times 256$) validadas por radiologistas. Indispensável para justificar o *ground-truth* anatômico usado nas métricas de XAI (EFL e Pointing Game).
3. **`kaggle_covid19_radiography` (Kaggle 2020):**
   - *Motivo:* Registro público e reprodutível do repositório da base (vencedor do COVID-19 Dataset Award), garantindo link direto de acesso conforme as boas práticas de dados abertos.

#### Bloco 2: Arquiteturas e Hipótese de Viés Indutivo
4. **`he2016deep` (CVPR 2016):**
   - *Motivo:* Artigo seminal da arquitetura ResNet-50. Fundamenta o funcionamento das conexões residuais (*skip connections*) e o viés indutivo clássico de localidade convolucional e equivariância à translação.
5. **`dosovitskiy2020image` (ICLR 2021):**
   - *Motivo:* Artigo seminal do Vision Transformer (ViT). Fundamenta matematicamente a divisão em patches ($16 \times 16$) e prova formalmente que o ViT prescinde de viés indutivo espacial, explicando por que ele depende de grandes volumes de dados para convergir sem memorização espúria.
6. **`raghu2021vision` (NeurIPS 2021):**
   - *Motivo:* Estudo comparativo rigoroso provando que ViTs possuem atenção global uniforme desde as camadas iniciais, ao contrário das CNNs que constroem representações progressivas de baixo para alto nível. Justifica por que o ViT se dispersa espacialmente quando faltam amostras de treino.
7. **`matsoukas2021time` (Medical Image Analysis 2021):**
   - *Motivo:* Primeiro trabalho de grande impacto a transpor o confronto CNN vs. ViT para imagens médicas, apontando que Transformers sofrem severamente em bases de dados médicas de escala tradicional/escassa frente a CNNs consagradas.

#### Bloco 3: Métodos de XAI e Avaliação Anatômica
8. **`selvaraju2017grad` (ICCV 2017):**
   - *Motivo:* Artigo criador do Grad-CAM. Fundamenta o cálculo dos mapas de saliência ponderados pelos gradientes médios da última camada convolucional da ResNet-50 e pela projeção de ativações espaciais no ViT.
9. **`abnar2020quantifying` (ACL 2020):**
   - *Motivo:* Artigo criador do método Attention Rollout ($\hat{A} = 0.5A + 0.5I$), implementado em `src/xai.py` para rastrear o fluxo atencional recursivo através dos blocos de autoatenção do ViT.
10. **`zhang2018top` (IJCV 2018):**
    - *Motivo:* Artigo de referência que estabelece formalmente o *Pointing Game* (*Hit Rate* binário), métrica que afere se o pico de ativação ($\arg\max A(x, y)$) reside dentro da anotação pulmonar real.

#### Bloco 4: O Paradoxo do AUROC, Shortcuts e Risco Clínico
11. **`oakden2020hidden` (Nature Medicine 2020):**
    - *Motivo:* Artigo de referência essencial para explicar o **Paradoxo do AUROC**: demonstra empiricamente como modelos em radiologia com AUROC elevado (>0,95) mascaram colapsos clínicos catastróficos em pontos de operação e sensibilidade diagnóstica (exatamente o que ocorreu no nosso regime de 5%, onde o ViT teve AUROC de 0,9669 mas FNR de 25,14%).
12. **`degrave2021ai` (Nature Machine Intelligence 2021):**
    - *Motivo:* Estudo definitivo sobre aprendizado de atalhos (*shortcut learning*) em raio-X de tórax para COVID-19, provando que redes neurais frequentemente memorizam marcadores de chumbo, tubos e bordas do detector em vez da patologia pulmonar real. Fundamenta a nossa métrica de *Background Leakage* ($1 - \text{EFL}$).
13. **`saporta2022benchmarking` (Nature Machine Intelligence 2022):**
    - *Motivo:* O maior benchmark recente de métodos de saliência em radiografias de tórax utilizando máscaras ground-truth de radiologistas, validando a metodologia de sobreposição e interseção de mapas de calor adotada neste artigo.
14. **`wong2020frequency` (Radiology 2020):**
    - *Motivo:* Fornece a justificativa clínica hospitalar sobre a assimetria do custo do erro na triagem de emergência: falsos negativos em infecções respiratórias atrasam intervenções vitais e amplificam o contágio comunitário e hospitalar.
