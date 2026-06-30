# Selective Risk Framework

Esse repositório contém a implementação do framework de avaliação e mitigação de risco em classificação seletiva e detecção de dados fora de distribuição (OOD - Out-of-Distribution). O framework integra o método **LAQDA** (Label-Aware Quantitative Data Analysis), algoritmos de **Baselines** (MSP, Energy Score, Distância de Mahalanobis, kNN) e o controlador de rejeição controlada **SGR** (Selection with Guaranteed Risk).

---

## Estrutura do Framework

A estrutura do projeto está organizada de forma modular:

*   **`configs/`**: Arquivos de configuração centralizados do modelo (ex: `model_config.yaml`).
*   **`data/`**: Componentes de dados, datamodules do PyTorch Lightning/customizados e samplers episódicos de k-shot.
*   **`methods/`**:
    *   **`laqda/`**: Módulos e executáveis do LAQDA (Label-Aware Encoder, QDA Sampler, Loss contrastiva, cli de treino/inferência).
    *   **`baselines/`**: Classificadores padrão e scorers probabilísticos (Maximum Softmax Probability - MSP, Energy Score), bem como métricas baseadas em representações latentes (Distância de Mahalanobis e k-Nearest Neighbors - kNN dentro de `distance/`).
    *   **`sgr/`**: O algoritmo **SGR** (Selection with Guaranteed Risk) para controle estatístico de risco e limiarização pós-hoc.
    *   **`metrics/`**: A suíte unificada de avaliação (Acurácia, F1-Score, ECE calibrado, AUROC, FPR@95, AUPR e AURC).
*   **`tests/`**: Testes automatizados para validação matemática e de integridade dos componentes do framework.

---

## Configuração do Ambiente

O framework suporta duas formas principais de execução: nativa via `uv` (gerenciador rápido de pacotes Python) ou containerizada via `Docker`.

### Opção A: Execução Nativa (Recomendado para Cluster/DGX)

O projeto utiliza o `uv` para automatizar e acelerar a gerência de pacotes virtuais.

1.  **Instalar o UV e Sincronizar Dependências**:
    Na raiz do projeto, inicialize o ambiente virtual integrado:
    ```bash
    make setup-env
    ```
    *(Isso instalará a ferramenta `uv` e gerará automaticamente uma pasta oculta `.venv` com todas as dependências especificadas instaladas).*

2.  **Ativar o Ambiente Virtual**:
    Antes de rodar qualquer script do framework, ative o ambiente virtual criado:
    ```bash
    source .venv/bin/activate
    ```

### Opção B: Execução Containerizada (Docker)

Se você preferir executar o treinamento de forma isolada usando contêineres com suporte a GPU:

1.  **Construir a Imagem**:
    ```bash
    make laqda-install
    ```
    *(Roda internamente o comando `docker compose build laqda`).*

2.  **Subir o Container com GPU**:
    ```bash
    docker compose up laqda
    ```

---

## Como Executar

### 1. Treinamento do LAQDA

O LAQDA pode ser treinado com ou sem a ativação da otimização do threshold SGR.

*   **Treinamento Padrão**:
    ```bash
    python -m methods.laqda.cli.train \
        --train_file datasets-br-nlp/train.jsonl \
        --valid_file datasets-br-nlp/valid.jsonl
    ```
*   **Treinamento Integrado com SGR**:
    Estima e salva o threshold pós-hoc correspondente a **5% de risco** na validação.
    ```bash
    python -m methods.laqda.cli.train \
        --train_file datasets-br-nlp/train.jsonl \
        --valid_file datasets-br-nlp/valid.jsonl \
        --use_sgr
    ```

### 2. Treinamento de Baselines

Para rodar os métodos MSP, Energy, Mahalanobis e kNN, primeiro treine o classificador supervisionado base:
```bash
python -m methods.baselines.cli.train_baseline \
    --train_file datasets-br-nlp/train.jsonl \
    --valid_file datasets-br-nlp/valid.jsonl
```

### 3. Executando Inferências e Avaliação (LAQDA)

Para submissões CodaBench ou testes em múltiplos folds com ensemble:
```bash
python -m methods.laqda.cli.infer \
    --train_file datasets-br-nlp/train.jsonl \
    --test_file datasets-br-nlp/test.jsonl \
    --model_paths outputs/fold1.pth outputs/fold2.pth
```

### 4. Consolidando Métricas (LAQDA)

Após salvar os arquivos `result.csv` correspondentes a cada execução de inferência, você pode gerar um relatório pivotado:
```bash
make laqda-eval
```

---

## Testes Automatizados

O framework possui uma bateria de testes unitários para validar se os scorers OOD e de distância estão funcionando corretamente e gerando saídas nos formatos corretos.

Para executar os testes:
```bash
make test-baselines
```
*(Ou execute diretamente pelo interpretador: `python -m tests.test_metrics`).*

---

## Execução e Monitoramento via Cluster (SLURM)

Para rodar experimentos em segundo plano no cluster, use os scripts SLURM configurados.

*   **Submeter Job**:
    ```bash
    sbatch run_test.slurm
    ```
*   **Acompanhar Saída em Tempo Real**:
    ```bash
    tail -f laqda_saida_<JOB_ID>.log
    ```
*   **Monitorar Erros**:
    ```bash
    cat laqda_erro_<JOB_ID>.log
    ```
*   **Verificar Fila de Jobs**:
    ```bash
    squeue -u $USER
    ```
*   **Cancelar Job**:
    ```bash
    scancel <JOB_ID>
    ```

---

## Comandos do Makefile

O `Makefile` atua como o orquestrador de atalhos rápidos do framework:

| Comando | Descrição |
|:---|:---|
| `make setup-env` | Cria a `.venv` local e instala dependências via `uv`. |
| `make laqda-install` | Cria a imagem Docker para o LAQDA. |
| `make laqda-train` | Exibe a ajuda com parâmetros do script de treino do LAQDA. |
| `make laqda-infer` | Exibe exemplo de parâmetros do script de inferência do LAQDA. |
| `make train-baseline` | Exibe exemplo de treino do modelo baseline. |
| `make test-baselines` | Executa a suíte de testes unitários (`tests/test_metrics.py`). |
| `make run-pipeline` | Executa o script unificado (`run_all.sh`). |
| `make laqda-eval` | Consolida e apresenta os relatórios e métricas consolidadas. |
| `make laqda-lint` | Valida sintaxe e compilação dos módulos Python. |
| `make laqda-clean` | Remove arquivos temporários de cache e pastas `__pycache__`. |

---

## Matriz de Métricas Implementada

Todas as métricas especificadas abaixo são calculadas e exportadas automaticamente no fim de cada ciclo de teste:

| Eixo de Avaliação | Métrica | Definição / Objetivo |
|:---|:---|:---|
| **Classificação Base** | Acurácia & F1-Score | Mede o desempenho sobre as previsões aceitas pelo modelo. |
| **Calibração** | ECE (Expected Calibration Error) | Quantifica a discrepância entre a confiança e a precisão empírica. |
| **Separação OOD** | AUROC / FPR@95 | Avalia a separabilidade entre dados In-Distribution (ID) vs Out-of-Distribution (OOD). |
| **Separação OOD** | AUPR-IN / AUPR-OUT | Mede a separabilidade sob desbalanceamento severo de classes. |
| **Predição Seletiva** | Curva RC e AURC | Avalia o trade-off entre risco e cobertura acumulada. |
| **Predição Seletiva** | Risk @ Cobertura Fixo | Mede o risco quando a cobertura é travada em patamares como 80%, 90% ou 95%. |
| **Garantia de Risco** | Cobertura do SGR | Mensura a cobertura empírica alcançada ao fixar níveis toleráveis de risco. |
